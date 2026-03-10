import os
import pandas as pd
import numpy as np
import joblib # For loading the ML model
import itertools
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # For input validation
from typing import List, Optional, Dict
from difflib import get_close_matches # For fuzzy name matching

# --- RDKit Imports ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    print("RDKit imported successfully.")
    RDKIT_AVAILABLE = True
except ImportError:
    print("WARNING: RDKit not found. ML predictions will be disabled.")
    RDKIT_AVAILABLE = False

# --- Configuration ---
DB_PATH = "drugbank.db" # Path to your SQLite database
DRUGS_CSV_PATH = os.path.join("parsed_data", 'drugs.csv') # Source for startup loading
MODEL_FILE = "ddi_xgb_model.pkl" # Your trained XGBoost model file
ML_PREDICTION_THRESHOLD = 0.5 # Probability threshold to report ML prediction

# --- Global Variables for Loaded Data/Model ---
# Store only SELECTED drug info for LLM context {drugbank_id: {name: '...', smiles: '...', ...}}
id_to_drug_info: Dict[str, Dict] = {}
name_to_id: Dict[str, str] = {} # Keep this for quick name lookup
ml_model = None
fp_generator = None # RDKit fingerprint generator

# --- Database Helper Functions ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        return conn
    except sqlite3.Error as e:
        print(f"ERROR: Could not connect to database at {DB_PATH}: {e}")
        return None

def get_drug_id_by_name_sql(conn, name: str) -> Optional[str]:
    """Finds DrugBank ID by exact name, close name match, or synonym match in SQLite."""
    if not conn: return None
    cursor = conn.cursor()
    search_name_lower = name.lower()

    # 1. Try exact match on name (case-insensitive)
    cursor.execute("SELECT drugbank_id FROM drugs WHERE LOWER(name)=?", (search_name_lower,))
    match = cursor.fetchone()
    if match:
        return match["drugbank_id"]

    # 2. Try exact match within synonyms (case-insensitive, split by '; ')
    cursor.execute("""
        SELECT drugbank_id FROM drugs
        WHERE LOWER(synonyms) LIKE ?
           OR LOWER(synonyms) LIKE ?
           OR LOWER(synonyms) LIKE ?
           OR LOWER(synonyms) = ?
        """,
        (f'%; {search_name_lower}; %', # Match in the middle
         f'{search_name_lower}; %',   # Match at the beginning
         f'%; {search_name_lower}',   # Match at the end
         search_name_lower)           # Match if it's the only synonym
    )
    match = cursor.fetchone()
    if match:
        return match["drugbank_id"]

    # 3. Try close match on name if other searches fail
    cursor.execute("SELECT name FROM drugs WHERE name IS NOT NULL")
    all_names = [row["name"] for row in cursor.fetchall()]
    # Filter out None values before list comprehension
    all_names_lower = [n.lower() for n in all_names if n]
    close_matches = get_close_matches(search_name_lower, all_names_lower, n=1, cutoff=0.8)
    if close_matches:
        # We found a close match to a *primary name*, get its ID
        cursor.execute("SELECT drugbank_id FROM drugs WHERE LOWER(name)=?", (close_matches[0],))
        match = cursor.fetchone()
        return match["drugbank_id"] if match else None

    return None # No match found

def get_ddi_alerts_sql(conn, drug_ids: list, drug_info_map: dict) -> List[dict]:
    """Gets known DDI alerts from SQLite for pairs within the list."""
    alerts = []
    if not conn or len(drug_ids) < 2:
        return alerts

    cursor = conn.cursor()
    # Check all unique pairs
    for id1, id2 in itertools.combinations(drug_ids, 2):
        # Query in both directions (a->b and b->a)
        cursor.execute(
            """SELECT drug_a_id, drug_b_id, description
               FROM ddi_rules
               WHERE (drug_a_id = ? AND drug_b_id = ?) OR (drug_a_id = ? AND drug_b_id = ?)""",
            (id1, id2, id2, id1)
        )
        row = cursor.fetchone()
        if row:
            # Use id_to_drug_info mapping (loaded at startup) for better names
            name1 = drug_info_map.get(row["drug_a_id"], {}).get("name", row["drug_a_id"])
            name2 = drug_info_map.get(row["drug_b_id"], {}).get("name", row["drug_b_id"])
            alerts.append({
                "drug1_name": name1, "drug1_id": row["drug_a_id"],
                "drug2_name": name2, "drug2_id": row["drug_b_id"],
                "description": row["description"],
                "level": "HIGH", # Assuming all known DB DDIs are significant
                "source": "Rule Engine (DB - DDI)"
            })
    return alerts

def get_dfi_alerts_sql(conn, drug_ids: list, drug_info_map: dict) -> List[dict]:
    """Gets DFI notes from SQLite for each drug ID in the list."""
    alerts = []
    if not conn or not drug_ids:
        return alerts

    placeholders = ','.join('?' * len(drug_ids))
    query = f"SELECT drug_id, food_interaction FROM dfi_rules WHERE drug_id IN ({placeholders})"
    cursor = conn.cursor()
    cursor.execute(query, tuple(drug_ids))
    rows = cursor.fetchall()

    # Group notes by drug_id
    notes_by_drug = {}
    for row in rows:
        notes_by_drug.setdefault(row["drug_id"], []).append(row["food_interaction"])

    # Create alerts
    for drug_id, notes in notes_by_drug.items():
        full_note = " | ".join(n for n in notes if n) # Join multiple notes
        if full_note:
            drug_name = drug_info_map.get(drug_id, {}).get("name", drug_id) # Use loaded name map
            alerts.append({
                "drug_name": drug_name,
                "drug_id": drug_id,
                "note": full_note,
                "source": "DrugBank Data (DB - DFI)"
            })
    return alerts


# --- ML Helper Functions ---
def get_fingerprint(smiles: str):
    """Generates Morgan fingerprint from SMILES using the global generator."""
    if not RDKIT_AVAILABLE or fp_generator is None or not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        fp = fp_generator.GetFingerprint(mol)
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        # Option: Log detailed RDKit errors if needed
        # print(f"RDKit Error processing SMILES: {smiles[:50]}...")
        return None

# --- Startup Data/Model Loading ---
def load_app_data():
    """Loads ML model and SELECTED essential drug info from CSV into memory."""
    global ml_model, fp_generator, id_to_drug_info, name_to_id
    print("Loading ML model and drug info...")
    try:
        # Load Model
        if RDKIT_AVAILABLE and os.path.exists(MODEL_FILE):
            ml_model = joblib.load(MODEL_FILE)
            fp_generator = GetMorganGenerator(radius=2, fpSize=2048)
            print("  ML model loaded.")
        else:
            print(f"WARNING: Model file '{MODEL_FILE}' not found or RDKit unavailable. ML predictions disabled.")
            ml_model = None

        # Load Drug Info from CSV - SELECT specific columns for LLM context
        if os.path.exists(DRUGS_CSV_PATH):
            # === Define FOCUSED columns to load ===
            focused_cols = [
                'drugbank_id', # Needed as index
                'name',
                'smiles', # Needed for ML fingerprinting
                'atc_code',
                'categories',
                'drug_groups',
                'description' # Keep description
            ]
            # === End column selection ===

            # Use usecols to load only necessary columns, dtype to ensure string IDs
            drugs_df = pd.read_csv(DRUGS_CSV_PATH, usecols=lambda c: c in focused_cols,
                                   dtype={'drugbank_id': str, 'name': str}) # Force types early

            # Drop rows missing critical info AFTER loading subset
            drugs_df.dropna(subset=['drugbank_id', 'name'], inplace=True)
            # Ensure ID is string type AFTER potential reads as float/int
            drugs_df['drugbank_id'] = drugs_df['drugbank_id'].astype(str)

            # Fill NaN for easier processing in selected columns
            drugs_df.fillna("", inplace=True)

            # Store selected columns in id_to_drug_info
            id_to_drug_info = drugs_df.set_index('drugbank_id')[
                [col for col in focused_cols if col != 'drugbank_id'] # All selected except index
            ].to_dict(orient='index')

            # Create name_to_id mapping (still needed for lookup)
            name_to_id = drugs_df.set_index(drugs_df['name'].str.lower())['drugbank_id'].to_dict()

            print(f"  Loaded focused info for {len(id_to_drug_info)} drugs from CSV.")
            print(f"  Loaded {len(name_to_id)} name mappings.")
        else:
             print(f"FATAL ERROR: Drugs CSV '{DRUGS_CSV_PATH}' not found. Cannot load essential data.")
             id_to_drug_info = {}
             name_to_id = {}

    except Exception as e:
        print(f"FATAL ERROR during startup loading: {e}")
        ml_model = None
        id_to_drug_info = {}
        name_to_id = {}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Drug Interaction Checker API",
    description="Checks known (via DB) and predicts potential (via ML) drug interactions. Provides info for LLM synthesis."
)

@app.on_event("startup")
async def startup_event():
    # Ensure DB exists, if not, maybe run init_database() from rule_db.py?
    if not os.path.exists(DB_PATH):
         print(f"WARNING: Database {DB_PATH} not found. Rule checks will fail.")
         # Consider adding initialization if needed for deployment
         # from rule_db import init_database
         # print("Initializing database from CSVs...")
         # init_database()
         # print("Database initialized.")

    load_app_data() # Load model and drug info from CSV


# --- Input Model Definition ---
class InteractionCheckRequest(BaseModel):
    drug_names: List[str] = Field(..., min_length=1, example=["Warfarin", "Aspirin", "Simvastatin"])
    # Add optional context fields later for Gemini
    # patient_age_group: Optional[str] = None
    # patient_allergies: Optional[List[str]] = None
    # patient_diagnoses: Optional[List[str]] = None
    # food_query: Optional[str] = None

# --- API Endpoint (/check) ---
@app.post("/check", summary="Check Drug Interactions")
async def check_interactions(request: InteractionCheckRequest):
    """
    Receives drug names, checks DB for known DDIs/DFIs, predicts potential DDIs via ML.
    Returns structured data ready for LLM processing.
    """
    drug_names_input = request.drug_names
    # Initialize results structure (No category_warnings needed)
    results = {
        "input_drugs_processed": [], # Will contain focused info
        "known_ddi": [],
        "dfi_notes": [],
        "predicted_ddi": [],
        "warnings": []
    }

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database connection unavailable.")

    # 1. Map names to IDs and retrieve FOCUSED drug info from memory
    valid_drug_ids = []
    processed_drug_info_map = {} # {drug_id: {focused_info}}
    for name in drug_names_input:
        drug_id = get_drug_id_by_name_sql(conn, name) # Use DB for robust lookup
        if drug_id:
            if drug_id not in processed_drug_info_map: # Avoid duplicates
                # Retrieve the FOCUSED info dictionary loaded at startup
                drug_info = id_to_drug_info.get(drug_id)
                if drug_info:
                     valid_drug_ids.append(drug_id)
                     processed_drug_info_map[drug_id] = drug_info
                     # Add FOCUSED info to results
                     results["input_drugs_processed"].append({
                         "input_name": name,
                         "matched_id": drug_id,
                         **drug_info # Unpack focused details (name, smiles, atc, cats, groups, desc)
                     })
                else:
                    # This case should be rare if CSV loading worked
                    results["warnings"].append(f"Info for drug ID '{drug_id}' (from '{name}') not loaded into memory.")
        else:
            results["warnings"].append(f"Drug name '{name}' not found or matched in database.")

    # 2. Check Known DDIs (using DB, use names from processed_drug_info_map)
    results["known_ddi"] = get_ddi_alerts_sql(conn, valid_drug_ids, processed_drug_info_map)
    known_ddi_found_pairs = {frozenset([a["drug1_id"], a["drug2_id"]]) for a in results["known_ddi"]}

    # 3. Get DFI Notes (using DB, use names from processed_drug_info_map)
    results["dfi_notes"] = get_dfi_alerts_sql(conn, valid_drug_ids, processed_drug_info_map)

    conn.close() # Close DB connection

    # 4. Predict Potential DDIs (ML Model, using SMILES from processed_drug_info_map)
    if ml_model and RDKIT_AVAILABLE and len(valid_drug_ids) >= 2:
        print("\n--- Running ML Predictions ---")
        prediction_count = 0
        for id1, id2 in itertools.combinations(valid_drug_ids, 2):
            pair = frozenset([id1, id2])
            # Get SMILES from the processed info map
            smiles1 = processed_drug_info_map.get(id1, {}).get("smiles")
            smiles2 = processed_drug_info_map.get(id2, {}).get("smiles")

            # Predict only if NOT found by rule AND both have SMILES
            if pair not in known_ddi_found_pairs and smiles1 and smiles2:
                fp1 = get_fingerprint(smiles1)
                fp2 = get_fingerprint(smiles2)

                if fp1 is not None and fp2 is not None:
                    try:
                        features = np.concatenate([fp1, fp2]).reshape(1, -1).astype(np.float32)
                        probability = ml_model.predict_proba(features)[0][1] # Prob of class 1
                        prediction_count += 1

                        if probability >= ML_PREDICTION_THRESHOLD:
                            risk_level = "High (>=75%)" if probability >= 0.75 else f"Moderate ({(probability*100):.0f}%)"
                            name1 = processed_drug_info_map.get(id1, {}).get("name", id1)
                            name2 = processed_drug_info_map.get(id2, {}).get("name", id2)
                            results["predicted_ddi"].append({
                                "drug1_name": name1, "drug1_id": id1,
                                "drug2_name": name2, "drug2_id": id2,
                                "probability": float(probability), # Ensure JSON serializable
                                "predicted_level": risk_level,
                                "source": "ML Model (XGBoost)"
                            })
                    except Exception as e:
                         results["warnings"].append(f"Error during ML prediction for {id1}+{id2}: {e}")
        print(f"Ran {prediction_count} ML predictions.")

    # 5. TODO: Call Gemini API here
    # This 'results' dictionary now contains the focused drug info, known DDIs, DFI notes,
    # predicted DDIs, and warnings.
    # You would combine this with the optional user inputs (food query, patient context)
    # from the 'request' object and send it all to Gemini with your synthesis prompt.
    # Example:
    # user_context = {
    #     "food_query": request.food_query,
    #     "patient_diagnoses": request.patient_diagnoses
    #     # ... etc
    # }
    # gemini_report = call_gemini_synthesizer(results, user_context)
    # return {"report": gemini_report} # Return Gemini's output

    print("\n--- Sending Results (Focused, Raw - Pre-Gemini) ---")
    return results # Return raw results for now

# --- Run the app (for local development) ---
if __name__ == "__main__":
    import uvicorn
    # Make sure DB exists before starting
    if not os.path.exists(DB_PATH):
        print(f"Database file '{DB_PATH}' not found.")
        print("Please run `python rule_db.py` first to initialize the database.")
    else:
        # Run using: uvicorn main:app --reload --port 8000
        print("Starting FastAPI server on http://127.0.0.1:8000")
        print("Access API docs at http://127.0.0.1:8000/docs")
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
