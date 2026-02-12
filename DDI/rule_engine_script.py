import os
import pandas as pd
from experta import *
from difflib import get_close_matches
import json

# --- Configuration ---
DATA_DIR = "parsed_data"  # Directory containing your CSVs
DDI_FILE = os.path.join(DATA_DIR, 'ddi_rules.csv')
DFI_FILE = os.path.join(DATA_DIR, 'dfi_rules.csv')
DRUGS_FILE = os.path.join(DATA_DIR, 'drugs.csv')
# --- End Configuration ---


# --- Define Facts ---
class Drug(Fact):
    """Fact representing a drug input by the user."""
    id = Field(str, mandatory=True)
    name = Field(str, mandatory=True)


class Alert(Fact):
    """Fact representing a found interaction or warning."""
    level = Field(str, mandatory=True)  # HIGH, MEDIUM, INFO
    source = Field(str, mandatory=True)  # DDI Rule, DFI Info
    message = Field(str, mandatory=True)
    score = Field(int, mandatory=True)   # numerical severity (for sorting)


# --- Knowledge Engine ---
class InteractionChecker(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self._load_rules_data()

    def _load_rules_data(self):
        """Load DDI, DFI, and drug metadata from CSV files."""
        print("Loading DrugBank interaction data...")
        try:
            # --- Load DDI Rules ---
            ddi_df = pd.read_csv(DDI_FILE)
            self.ddi_pairs = set()
            self.ddi_descriptions = {}
            for _, row in ddi_df.iterrows():
                a, b = str(row['drug_a_id']), str(row['drug_b_id'])
                pair = frozenset([a, b])
                self.ddi_pairs.add(pair)
                if pair not in self.ddi_descriptions:
                    self.ddi_descriptions[pair] = row.get('description', 'Known interaction.')
            print(f"  Loaded {len(self.ddi_pairs)} DDI pairs.")

            # --- Load DFI Rules ---
            dfi_df = pd.read_csv(DFI_FILE)
            dfi_df['drug_id'] = dfi_df['drug_id'].astype(str)
            self.dfi_data = dfi_df.groupby('drug_id')['food_interaction'].apply(
                lambda x: ' | '.join(x.dropna().astype(str))
            ).to_dict()
            print(f"  Loaded DFI data for {len(self.dfi_data)} drugs.")

            # --- Load Drug Names ---
            drugs_df = pd.read_csv(DRUGS_FILE)
            drugs_df['drugbank_id'] = drugs_df['drugbank_id'].astype(str)
            self.drug_names = drugs_df.set_index('drugbank_id')['name'].to_dict()
            print(f"  Loaded names for {len(self.drug_names)} drugs.\n")

        except Exception as e:
            print(f"ERROR loading data: {e}")
            self._initialize_empty_data()

    def _initialize_empty_data(self):
        self.ddi_pairs = set()
        self.ddi_descriptions = {}
        self.dfi_data = {}
        self.drug_names = {}

    # --- Rules ---
    @Rule(AS.d1 << Drug(id=MATCH.id1),
          AS.d2 << Drug(id=MATCH.id2),
          TEST(lambda id1, id2: str(id1) < str(id2)))
    def known_ddi_found(self, id1, id2, d1, d2):
        """Detects known DDI pairs."""
        pair = frozenset([str(id1), str(id2)])
        if pair not in self.ddi_pairs:
            return
        description = self.ddi_descriptions.get(pair, "Known interaction.")
        message = f"Known DDI: {d1['name']} and {d2['name']}. Reason: {description}"
        self.declare(Alert(level='HIGH', source='DDI Rule', message=message, score=3))

    @Rule(AS.drug << Drug(id=MATCH.drug_id))
    def known_dfi_info(self, drug, drug_id):
        """Adds food interaction notes for each drug."""
        drug_id_str = str(drug_id)
        if drug_id_str not in self.dfi_data:
            return
        text = self.dfi_data.get(drug_id_str, "")
        if not text.strip():
            return

        # Basic severity heuristic
        lower = text.lower()
        if "avoid" in lower:
            level, score = 'HIGH', 2
        elif "take with" in lower or "reduce" in lower:
            level, score = 'MEDIUM', 1
        else:
            level, score = 'INFO', 0

        message = f"Food Interaction for {drug['name']}: {text}"
        self.declare(Alert(level=level, source='DFI Info', message=message, score=score))


# --- Fuzzy Drug Match ---
def find_drug_id(engine, input_name):
    """Finds the best DrugBank ID for a given name."""
    for db_id, db_name in engine.drug_names.items():
        if db_name and input_name.lower() == db_name.lower():
            return db_id

    possible = get_close_matches(input_name.lower(),
                                 [n.lower() for n in engine.drug_names.values()],
                                 cutoff=0.8)
    if possible:
        match_name = possible[0]
        for db_id, db_name in engine.drug_names.items():
            if db_name.lower() == match_name:
                return db_id
    return None


# --- Runner Function ---
def run_interaction_check(engine_instance, drug_names):
    """Runs checks for given drug list and returns structured alerts."""
    engine_instance.reset()
    declared_ids = set()

    print("\n--- Declaring Drug Facts ---")
    for name in drug_names:
        found_id = find_drug_id(engine_instance, name)
        if found_id:
            if found_id not in declared_ids:
                engine_instance.declare(Drug(id=str(found_id), name=name))
                declared_ids.add(found_id)
                print(f"Declared: {name} (ID: {found_id})")
        else:
            print(f"Warning: '{name}' not found in DrugBank — ignored.")

    print("\nRunning rule-based checks...")
    engine_instance.run()

    alerts = [f for f in engine_instance.facts.values() if isinstance(f, Alert)]
    alerts.sort(key=lambda x: x['score'], reverse=True)
    return alerts


# --- Example Execution ---
if __name__ == "__main__":
    checker = InteractionChecker()
    input_drugs = ["Warfarin", "Simvastatin", "Aspirin"]
    print(f"\n===== Checking {input_drugs} =====")
    results = run_interaction_check(checker, input_drugs)

    alerts_output = []
    if results:
        print("\n--- Interaction Results ---")
        for alert in results:
            alerts_output.append({
                "level": alert["level"],
                "source": alert["source"],
                "message": alert["message"]
            })
            print(f"[{alert['level']}] {alert['source']}: {alert['message']}")
    else:
        print("No known interactions or food notes found.")

    print("\n*Disclaimer: Use this tool as a decision aid. Always verify with a clinician.*")

    # Save JSON result (for web/LLM layer)
    with open("interaction_results.json", "w", encoding="utf-8") as f:
        json.dump(alerts_output, f, indent=2, ensure_ascii=False)
