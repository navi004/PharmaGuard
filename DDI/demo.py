import os
import sqlite3
import pandas as pd
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# === CONFIGURATION ===
DATA_DIR = "parsed_data"
DRUGS_FILE = os.path.join(DATA_DIR, 'drugs.csv')
DB_PATH = "drugbank.db"
MODEL_PATH = "ddi_xgb_model.pkl"   # Path to your trained RF model
TOP_N_RF = 10  # Number of top ML-predicted pairs to return

# --- Load SQLite rule-based info ---
drugs_df = pd.read_csv(DRUGS_FILE)
usable_drugs = drugs_df[drugs_df['smiles'].notna()][['drugbank_id','name','smiles']]
usable_drugs.set_index('drugbank_id', inplace=True)

# --- Load trained ML model ---
rf_model = joblib.load(MODEL_PATH)
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

def get_drug_id_by_name(name):
    # Exact match, then fuzzy lookup
    match = drugs_df[drugs_df['name'].str.lower() == name.lower()]
    if not match.empty:
        return match['drugbank_id'].values[0]
    # Fuzzy matching (for demonstration, returns best match if >80% similarity)
    from difflib import get_close_matches
    names = drugs_df['name'].tolist()
    match_name = get_close_matches(name, names, cutoff=0.8)
    if match_name:
        row = drugs_df[drugs_df['name'] == match_name[0]]
        return row['drugbank_id'].values[0]
    return None

def get_smiles_by_id(drug_id):
    return usable_drugs.loc[drug_id, 'smiles'] if drug_id in usable_drugs.index else None

def get_rule_ddi_alerts(ids):
    # Fetch all DDIs among given DrugBank IDs (bi-directional)
    if len(ids) < 2:
        return []
    conn = sqlite3.connect(DB_PATH)
    placeholders = ','.join(['?'] * len(ids))
    q = f"""
    SELECT drug_a_id, drug_b_id, drug_b_name, description FROM ddi_rules
    WHERE drug_a_id IN ({placeholders}) AND drug_b_id IN ({placeholders})
    """
    rows = conn.cursor().execute(q, tuple(ids + ids)).fetchall()
    conn.close()
    alerts = []
    for a_id, b_id, b_name, desc in rows:
        alerts.append({
            "type": "rule",
            "drugs": [a_id, b_id],
            "message": f"DDI (Rule): {a_id} ↔ {b_id} — {desc}",
            "score": 1,
            "source": "Rule-based"
        })
    # Remove duplicate alerts (A↔B vs B↔A)
    seen = set()
    unique_alerts = []
    for alert in alerts:
        pair = tuple(sorted(alert["drugs"]))
        if pair not in seen:
            unique_alerts.append(alert)
            seen.add(pair)
    return unique_alerts

def predict_rf_ddi(drugs_input_ids):
    # Find all pairs not covered by rules, run through ML model
    fingerprints = {}
    for dbid in drugs_input_ids:
        smi = get_smiles_by_id(dbid)
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = morgan_gen.GetFingerprint(mol)
                arr = np.zeros((1,))
                Chem.DataStructs.ConvertToNumpyArray(fp, arr)
                fingerprints[dbid] = arr
    # Generate all possible unique pairs
    pairs = []
    n = len(fingerprints)
    ids = list(fingerprints.keys())
    for i in range(n):
        for j in range(i+1, n):
            a, b = ids[i], ids[j]
            pairs.append((a, b))
    results = []
    for a, b in pairs:
        fp_a, fp_b = fingerprints[a], fingerprints[b]
        features = np.concatenate([fp_a, fp_b]).reshape(1, -1)
        prob = rf_model.predict_proba(features)[0][1]
        results.append({
            "type": "ml",
            "drugs": [a, b],
            "prob": float(prob),
            "message": f"Predicted DDI (ML): {a} ↔ {b}, Probability = {prob:.2f}",
            "risk_level": "High" if prob >= 0.75 else "Medium" if prob >= 0.5 else "Low",
            "source": "RandomForest"
        })
    # Return top-N highest probability pairs
    results_sorted = sorted(results, key=lambda x: x['prob'], reverse=True)
    return results_sorted[:TOP_N_RF]

def check_interactions(drug_names):
    # Main integration logic
    ids = []
    declared = []
    for name in drug_names:
        dbid = get_drug_id_by_name(name)
        if dbid:
            ids.append(dbid)
            declared.append({"name": name, "drugbank_id": dbid})
        else:
            declared.append({"name": name, "drugbank_id": None, "warning": "Drug not found"})
    rule_alerts = get_rule_ddi_alerts(ids)
    # Find all pairs present in input
    covered_pairs = {tuple(sorted(alert["drugs"])) for alert in rule_alerts}
    # ML on all uncovered pairs
    uncovered_ids = [d for d in ids if d]  
    ml_alerts = []
    if len(uncovered_ids) > 1:
        all_pairs = set(tuple(sorted([uncovered_ids[i], uncovered_ids[j]]))
                        for i in range(len(uncovered_ids)) for j in range(i+1, len(uncovered_ids)))
        ml_pairs = [p for p in all_pairs if p not in covered_pairs]
        # Prepare inputs for ML
        fingerprints = {dbid: get_smiles_by_id(dbid) for dbid in uncovered_ids}
        ml_alerts = predict_rf_ddi(uncovered_ids)
    # Combine results
    results = {
        "declared": declared,
        "rule_alerts": rule_alerts,
        "ml_alerts": ml_alerts
    }
    return results

# Example usage:
if __name__ == "__main__":
    # User input (example)
    input_drugs = ["Warfarin", "Simvastatin", "Aspirin"]
    results = check_interactions(input_drugs)
    print("\nDRUGS DECLARED:")
    for d in results["declared"]:
        print(d)
    print("\nRULE-BASED DDIs FOUND:")
    for r in results["rule_alerts"]:
        print(r["message"])
    print(f"\nTOP {TOP_N_RF} ML-PREDICTED DDIs:")
    for ml in results["ml_alerts"]:
        print(f"{ml['message']} (Risk: {ml['risk_level']}) [Probability: {ml['prob']:.2f}]")
