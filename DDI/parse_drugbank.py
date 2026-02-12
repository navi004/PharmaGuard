import xml.etree.ElementTree as ET
import csv
import os
from tqdm import tqdm

# === CONFIG ===
XML_PATH = r'C:\Users\nvkaj\OneDrive - vit.ac.in\sem 7\Health care Analytics\J component\drugbank_all_full_database.xml\full database.xml'
OUTPUT_DIR = "parsed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === PARSE XML ===
tree = ET.parse(XML_PATH)
root = tree.getroot()
ns = {'db': 'http://www.drugbank.ca'}

drugs_data = []
ddi_data = []
dfi_data = []

print("Parsing DrugBank XML...")

for drug in tqdm(root.findall('db:drug', ns)):
    try:
        db_id = drug.find('db:drugbank-id[@primary="true"]', ns).text
        name = drug.find('db:name', ns).text
        desc = drug.find('db:description', ns)
        desc = desc.text if desc is not None else ""

        # --- SMILES ---
        smiles = None
        props = drug.find('db:calculated-properties', ns)
        if props is not None:
            for prop in props.findall('db:property', ns):
                kind = prop.find('db:kind', ns)
                if kind is not None and kind.text == "SMILES":
                    smiles = prop.find('db:value', ns).text
                    break

        # --- ATC Code ---
        atc_elem = drug.find('db:atc-codes/db:atc-code', ns)
        atc_code = atc_elem.attrib.get('code') if atc_elem is not None else None

        # --- Synonyms ---
        syn_elems = drug.findall('db:synonyms/db:synonym', ns)
        synonyms = "; ".join([s.text for s in syn_elems if s.text])

        # Save drug info
        drugs_data.append({
            "drugbank_id": db_id,
            "name": name,
            "synonyms": synonyms,
            "smiles": smiles,
            "atc_code": atc_code,
            "description": desc
        })

        # --- DDI Extraction ---
        for ddi in drug.findall('db:drug-interactions/db:drug-interaction', ns):
            inter_id = ddi.find('db:drugbank-id', ns)
            inter_name = ddi.find('db:name', ns)
            desc_text = ddi.find('db:description', ns)
            if inter_id is not None and desc_text is not None:
                ddi_data.append({
                    "drug_a_id": db_id,
                    "drug_b_id": inter_id.text,
                    "drug_b_name": inter_name.text if inter_name is not None else "",
                    "description": desc_text.text
                })

        # --- DFI Extraction ---
        for food in drug.findall('db:food-interactions/db:food-interaction', ns):
            if food.text:
                dfi_data.append({
                    "drug_id": db_id,
                    "drug_name": name,
                    "food_interaction": food.text
                })

    except Exception as e:
        print(f"Error parsing drug: {e}")
        continue

# === SAVE TO CSV ===
with open(os.path.join(OUTPUT_DIR, 'drugs.csv'), 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(drugs_data[0].keys()))
    writer.writeheader()
    writer.writerows(drugs_data)

with open(os.path.join(OUTPUT_DIR, 'ddi_rules.csv'), 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(ddi_data[0].keys()))
    writer.writeheader()
    writer.writerows(ddi_data)

with open(os.path.join(OUTPUT_DIR, 'dfi_rules.csv'), 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(dfi_data[0].keys()))
    writer.writeheader()
    writer.writerows(dfi_data)

print("\n✅ Parsing complete!")
print(f"Drugs: {len(drugs_data)}, DDIs: {len(ddi_data)}, DFIs: {len(dfi_data)}")
print(f"Files saved to: {OUTPUT_DIR}/")
