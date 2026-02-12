import os
import pandas as pd
from experta import *

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
    level = Field(str, mandatory=True)  # e.g., HIGH, INFO
    source = Field(str, mandatory=True)  # e.g., DDI Rule, DFI Info
    message = Field(str, mandatory=True)


# --- The Rule Engine ---
class InteractionChecker(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self._load_rules_data()

    def _load_rules_data(self):
        """Loads DDI, DFI, and drug name data from CSV files."""
        print("Loading interaction data...")
        try:
            # --- Load DDI Rules ---
            ddi_df = pd.read_csv(DDI_FILE)
            self.ddi_pairs = set()
            self.ddi_descriptions = {}  # {frozenset({id1, id2}): description}
            for _, row in ddi_df.iterrows():
                id_a = str(row['drug_a_id'])
                id_b = str(row['drug_b_id'])
                pair = frozenset([id_a, id_b])
                self.ddi_pairs.add(pair)
                if pair not in self.ddi_descriptions:
                    self.ddi_descriptions[pair] = row.get('description', 'Known interaction.')
            print(f"  Loaded {len(self.ddi_pairs)} unique DDI pairs.")

            # --- Load DFI Rules ---
            dfi_df = pd.read_csv(DFI_FILE)
            dfi_df['drug_id'] = dfi_df['drug_id'].astype(str)
            # Group DFI text per drug
            self.dfi_data = dfi_df.groupby('drug_id')['food_interaction'].apply(
                lambda x: ' | '.join(x.dropna().astype(str))
            ).to_dict()
            print(f"  Loaded DFI information for {len(self.dfi_data)} drugs.")

            # --- Load Basic Drug Info ---
            drugs_df = pd.read_csv(DRUGS_FILE)
            drugs_df['drugbank_id'] = drugs_df['drugbank_id'].astype(str)
            self.drug_names = drugs_df.set_index('drugbank_id')['name'].to_dict()
            print(f"  Loaded names for {len(self.drug_names)} drugs.")

        except FileNotFoundError as e:
            print(f"ERROR: Could not find data file: {e}. Make sure CSVs are in '{DATA_DIR}'")
            self._initialize_empty_data()
        except pd.errors.EmptyDataError as e:
            print(f"ERROR: CSV file is empty: {e}. Check the parsing script output.")
            self._initialize_empty_data()
        except KeyError as e:
            print(f"ERROR: Missing expected column in CSV file: {e}. Check CSV headers.")
            self._initialize_empty_data()
        except Exception as e:
            print(f"ERROR loading rules data: {e}")
            self._initialize_empty_data()

    def _initialize_empty_data(self):
        """Sets data structures to empty defaults."""
        self.ddi_pairs = set()
        self.ddi_descriptions = {}
        self.dfi_data = {}
        self.drug_names = {}

    # --- Rule Definitions ---

    @Rule(AS.d1 << Drug(id=MATCH.id1),
          AS.d2 << Drug(id=MATCH.id2),
          TEST(lambda id1, id2: str(id1) < str(id2)))  # Check each pair only once
    def known_ddi_found(self, id1, id2, d1, d2):
        """Rule fires when a known DDI pair is declared."""
        pair = frozenset([str(id1), str(id2)])
        if pair not in self.ddi_pairs:
            return
        description = self.ddi_descriptions.get(pair, "Known interaction.")
        message = f"Known DDI: {d1['name']} ({id1}) and {d2['name']} ({id2}). Reason: {description}"
        self.declare(Alert(level='HIGH', source='DDI Rule', message=message))

    @Rule(AS.drug << Drug(id=MATCH.drug_id))
    def known_dfi_info(self, drug, drug_id):
        """Rule fires for each declared drug that has known DFI notes."""
        drug_id_str = str(drug_id)
        if drug_id_str not in self.dfi_data:
            return
        interaction_text = self.dfi_data.get(drug_id_str, "")
        if interaction_text.strip():
            message = f"Food Interaction Note for {drug['name']} ({drug_id_str}): {interaction_text}"
            self.declare(Alert(level='INFO', source='DFI Info', message=message))


# --- Function to Run Checks ---
def run_interaction_check(engine_instance: InteractionChecker, drug_names: list):
    """
    Runs the interaction check for a list of drug names using the provided engine instance.
    Returns a list of found alerts (DDIs and DFI info).
    """
    engine_instance.reset()

    declared_drug_ids = set()
    print("\n--- Declaring Drug Facts ---")

    for name in drug_names:
        found_id = None
        for db_id, db_name in engine_instance.drug_names.items():
            if db_name and name and db_name.lower() == name.lower():
                found_id = db_id
                break
        if found_id:
            found_id_str = str(found_id)
            if found_id_str not in declared_drug_ids:
                engine_instance.declare(Drug(id=found_id_str, name=name))
                declared_drug_ids.add(found_id_str)
                print(f"Declared Drug: {name} (ID: {found_id_str})")
        else:
            print(f"Warning: Could not find DrugBank ID for '{name}'. It will be ignored.")

    print("\nRunning interaction checks...")
    engine_instance.run()

    alerts = [fact for fact in engine_instance.facts.values() if isinstance(fact, Alert)]
    alerts.sort(key=lambda x: {'HIGH': 0, 'INFO': 1}.get(x['level'], 2))
    return alerts


# --- Example Usage ---
if __name__ == "__main__":
    checker = InteractionChecker()

    input_drugs_1 = ["Warfarin", "Simvastatin", "Aspirin"]
    print(f"\n===== CHECK 1: Drugs={input_drugs_1} =====")
    results_1 = run_interaction_check(checker, input_drugs_1)

    print("\n--- Interaction Results 1 ---")
    if not results_1:
        print("No known interactions or food notes found.")
    else:
        ddi_alerts = [a for a in results_1 if a['source'] == 'DDI Rule']
        dfi_alerts = [a for a in results_1 if a['source'] == 'DFI Info']

        print("\n** Drug-Drug Interactions (Known): **")
        if not ddi_alerts:
            print("None found.")
        else:
            for alert in ddi_alerts:
                print(f"- {alert['message']}")

        print("\n** Food Interaction Notes (General Advice): **")
        if not dfi_alerts:
            print("None found.")
        else:
            for alert in dfi_alerts:
                drug_name_part = alert['message'].split(':')[0]
                advice_part = alert['message'].split(':', 1)[1].strip()
                print(f"- {drug_name_part}:\n    {advice_part}")

        print("\n*Disclaimer: Consult your healthcare provider before making any changes to medication or diet.*")
