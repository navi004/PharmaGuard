import os
import pandas as pd
import sqlite3
from difflib import get_close_matches
import json
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

DATA_DIR = "parsed_data"
DB_PATH = "drugbank.db"
DRUGS_FILE = os.path.join(DATA_DIR, 'drugs.csv')
DDI_FILE = os.path.join(DATA_DIR, 'ddi_rules.csv')
DFI_FILE = os.path.join(DATA_DIR, 'dfi_rules.csv')


def init_database():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"{Fore.YELLOW}Old {DB_PATH} removed.{Style.RESET_ALL}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE drugs (
        drugbank_id TEXT PRIMARY KEY,
        name TEXT,
        synonyms TEXT,
        smiles TEXT,
        atc_code TEXT,
        description TEXT,
        drug_groups TEXT,
        categories TEXT,
        kingdom TEXT,
        superclass TEXT,
        class TEXT,
        subclass TEXT,
        target_ids TEXT,
        enzyme_ids TEXT,
        carrier_ids TEXT,
        transporter_ids TEXT,
        indication TEXT,
        mechanism_of_action TEXT,
        metabolism TEXT,
        toxicity TEXT,
        half_life TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE ddi_rules (
        drug_a_id TEXT,
        drug_b_id TEXT,
        drug_b_name TEXT,
        description TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE dfi_rules (
        drug_id TEXT,
        drug_name TEXT,
        food_interaction TEXT
    )
    """)

    print(f"{Fore.CYAN}Loading CSV data into SQLite...{Style.RESET_ALL}")
    pd.read_csv(DRUGS_FILE).to_sql('drugs', conn, if_exists='append', index=False)
    pd.read_csv(DDI_FILE).to_sql('ddi_rules', conn, if_exists='append', index=False)
    pd.read_csv(DFI_FILE).to_sql('dfi_rules', conn, if_exists='append', index=False)

    conn.commit()
    conn.close()
    print(f"{Fore.GREEN}Database initialized at: {DB_PATH}{Style.RESET_ALL}")


def get_drug_id_by_name(conn, name):
    cursor = conn.cursor()
    cursor.execute("SELECT drugbank_id FROM drugs WHERE LOWER(name)=?", (name.lower(),))
    match = cursor.fetchone()
    if match:
        return match[0]
    cursor.execute("SELECT name FROM drugs")
    all_names = [row[0] for row in cursor.fetchall() if row[0]]
    match_name = get_close_matches(name.lower(), [n.lower() for n in all_names], cutoff=0.8)
    if match_name:
        cursor.execute("SELECT drugbank_id FROM drugs WHERE LOWER(name)=?", (match_name[0],))
        d = cursor.fetchone()
        return d[0] if d else None
    return None


def get_ddi_alerts(conn, ids):
    if len(ids) < 2:
        return []
    placeholders = ','.join(['?'] * len(ids))
    q = f"""
    SELECT * FROM ddi_rules
    WHERE drug_a_id IN ({placeholders}) AND drug_b_id IN ({placeholders})
    """
    cursor = conn.cursor()
    rows = cursor.execute(q, tuple(ids + ids)).fetchall()

    alerts = []
    for a_id, b_id, b_name, desc in rows:
        alerts.append({
            "level": "HIGH",
            "source": "DDI Rule",
            "message": f"Known DDI: {a_id} ↔ {b_id}. {desc}"
        })
    return alerts


def get_dfi_alerts(conn, ids):
    if not ids:
        return []
    placeholders = ','.join(['?'] * len(ids))
    q = f"SELECT drug_name, food_interaction FROM dfi_rules WHERE drug_id IN ({placeholders})"
    cursor = conn.cursor()
    rows = cursor.execute(q, tuple(ids)).fetchall()

    alerts = []
    for drug_name, text in rows:
        lower = text.lower()
        if "avoid" in lower:
            level = "HIGH"
        elif "take with" in lower or "reduce" in lower:
            level = "MEDIUM"
        else:
            level = "INFO"

        alerts.append({
            "level": level,
            "source": "DFI Info",
            "message": f"Food Interaction Note for {drug_name}: {text}"
        })
    return alerts


def run_sql_interaction_check(drugs):
    conn = sqlite3.connect(DB_PATH)
    declared_ids = []

    print(f"\n{Fore.CYAN}--- Declaring Input Drugs ---{Style.RESET_ALL}")
    for name in drugs:
        found_id = get_drug_id_by_name(conn, name)
        if found_id:
            declared_ids.append(found_id)
            print(f"{Fore.GREEN}✔ Declared: {name} ({found_id}){Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}⚠ Could not find '{name}' in DrugBank.{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}--- Checking for Interactions ---{Style.RESET_ALL}")
    ddi_alerts = get_ddi_alerts(conn, declared_ids)
    dfi_alerts = get_dfi_alerts(conn, declared_ids)
    conn.close()

    # Remove duplicate DDIs
    unique_ddi = {}
    for alert in ddi_alerts:
        msg = alert["message"]
        drug_pair = frozenset(msg.split("↔"))
        if drug_pair not in unique_ddi:
            unique_ddi[drug_pair] = alert
    ddi_alerts = list(unique_ddi.values())

    # Group DFIs by drug name
    grouped_dfi = {}
    for alert in dfi_alerts:
        msg = alert["message"]
        if "for " in msg:
            drug_name = msg.split("for ")[1].split(":")[0].strip()
        else:
            drug_name = "Unknown"
        grouped_dfi.setdefault(drug_name, []).append(msg.split(":", 1)[1].strip())

    print(f"\n{Fore.YELLOW}=== Interaction Results ==={Style.RESET_ALL}")

    print(f"\n{Fore.MAGENTA}** Drug-Drug Interactions (Known): **{Style.RESET_ALL}")
    if not ddi_alerts:
        print("  None found.")
    else:
        for a in ddi_alerts:
            print(f"  {Fore.RED}- {a['message']}{Style.RESET_ALL}")

    print(f"\n{Fore.MAGENTA}** Food Interaction Notes: **{Style.RESET_ALL}")
    if not grouped_dfi:
        print("  None found.")
    else:
        for drug, notes in grouped_dfi.items():
            print(f"  {Fore.BLUE}{drug}:{Style.RESET_ALL}")
            for note in notes:
                print(f"    {Fore.CYAN}- {note}{Style.RESET_ALL}")

    print(f"\n{Fore.LIGHTBLACK_EX}*Disclaimer: Always consult a healthcare professional before altering medication or diet.*{Style.RESET_ALL}")

    structured = []
    for ddi in ddi_alerts:
        structured.append(ddi)
    for drug, notes in grouped_dfi.items():
        structured.append({
            "source": "DFI Info",
            "level": "HIGH",
            "drug": drug,
            "notes": notes
        })

    with open("interaction_results_structured.json", "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
        print(f"\n{Fore.GREEN}Results saved to 'interaction_results_structured.json'{Style.RESET_ALL}")


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        init_database()

    user_drugs = ["Warfarin", "Simvastatin", "Aspirin"]
    run_sql_interaction_check(user_drugs)
