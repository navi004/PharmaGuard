# PharmaGuard 💊
### Hybrid Drug Interaction Intelligence System

PharmaGuard is a hybrid AI system for predicting Drug-Drug Interactions (DDI) and Drug-Food Interactions (DFI). It combines rule-based expert reasoning with machine learning on molecular fingerprints, backed by a production-ready FastAPI backend and context-aware explanations via the Gemini API.

---

## Overview

Adverse drug interactions are a leading cause of preventable harm in clinical settings. PharmaGuard addresses this by fusing two complementary approaches:

- **Rule-based inference** using an expert system (Experta) that encodes known pharmacological rules
- **ML-based prediction** using molecular fingerprints (RDKit) with ensemble classifiers trained on 100K+ drug pairs

This hybrid design gives the system both the interpretability of expert rules and the predictive power of machine learning.

---

## Architecture

```
Input (Drug Pair / Drug + Food)
        │
        ├──► Rule Engine (Experta)
        │         └── Known interaction rules → Flags & severity levels
        │
        ├──► ML Pipeline (RDKit + Ensemble Models)
        │         ├── Morgan fingerprint generation
        │         ├── Random Forest classifier
        │         └── XGBoost classifier → Probability scores
        │
        └──► Fusion Layer
                  ├── Combined prediction output
                  ├── Gemini API → Context-aware natural language explanation
                  └── FastAPI → REST endpoint for real-time analysis
```

---

## Features

- **Hybrid prediction** — rule-based + ML ensemble for robust DDI/DFI detection
- **Molecular fingerprinting** — RDKit Morgan fingerprints encode drug structure
- **Ensemble models** — Random Forest + XGBoost trained on 100K drug pairs
- **Expert system** — Experta knowledge base with pharmacological interaction rules
- **LLM explanations** — Gemini API generates human-readable clinical context for each prediction
- **REST API** — modular FastAPI backend for real-time queries
- **Modular codebase** — OOP design patterns throughout for easy extension

---

## Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.94 |
| Accuracy | 0.88 |
| Dataset | 100,000 drug pairs |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Molecular features | RDKit |
| Rule engine | Experta |
| ML models | Scikit-learn, XGBoost |
| LLM explanations | Gemini API |
| Backend | FastAPI |
| Language | Python |


---

## Getting Started

### Prerequisites
```bash
Python 3.9+
```

### Installation
```bash
git clone https://github.com/navi004/Drug-Drug-Interaction.git
cd Drug-Drug-Interaction
pip install -r requirements.txt
```

### Run the API
```bash
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"drug_a": "warfarin", "drug_b": "aspirin"}'
```

### Example Response
```json
{
  "interaction": true,
  "severity": "high",
  "confidence": 0.91,
  "explanation": "Warfarin and aspirin both inhibit platelet aggregation through different mechanisms. Co-administration significantly increases bleeding risk."
}
```

---

## How It Works

1. **Input** — a drug pair (or drug + food item) is passed to the system
2. **Fingerprinting** — RDKit generates Morgan fingerprints encoding the molecular structure of each drug
3. **Rule check** — the Experta engine checks the pair against known pharmacological rules
4. **ML prediction** — the ensemble model outputs an interaction probability
5. **Fusion** — rule flags and ML scores are combined into a final prediction
6. **Explanation** — Gemini API generates a plain-language explanation of the interaction mechanism

---

## Future Work

- Expand knowledge base with additional DFI rules
- Add SMILES-based input for novel/unlisted compounds
- Build a clinical-facing web UI
- Integrate with drug databases (DrugBank, PubChem)

---

## Author

**Naveen Nidadavolu**  
[GitHub](https://github.com/navi004) • [LinkedIn](https://linkedin.com/in/naveen-nidadavolu-482254250)
