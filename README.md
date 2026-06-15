# 🔍 Support Integrity Auditor (SIA)
### MARS Open Projects 2026 — Part I, Problem Statement 1

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sia-app-s5rurfnamkoupwehaxnqyy.streamlit.app)

> A semantics-driven, evidence-grounded automated auditor that detects **Priority Mismatch** in customer support tickets — catching *Hidden Crises* (under-prioritised) and *False Alarms* (over-prioritised).

---

## 🌐 Live Demo
**Streamlit App:** https://sia-app-s5rurfnamkoupwehaxnqyy.streamlit.app

Supports:
- **Single Ticket Test** — paste one ticket, get instant verdict + Evidence Dossier
- **Batch CSV Test** — upload CSV of tickets, download full results + dossiers JSON

---

## 🏗️ Architecture

```
Raw Ticket Data
      │
      ▼
┌─────────────────────────────────────────────┐
│           STAGE 1: Pseudo-Label Generation  │
│                                             │
│  Signal 1 ── Rule-Based NLP (weight: 0.50) │
│              keyword density, negation,     │
│              escalation phrases             │
│                                             │
│  Signal 2 ── Sentence Embeddings (0.30)    │
│              all-MiniLM-L6-v2 + KMeans(4)  │
│                                             │
│  Signal 3 ── Resolution Time (0.20)        │
│              quartile-based bucketing       │
│                                             │
│  → Fused Severity Score                     │
│  → Binary Mismatch Label (delta ≥ 2)        │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│      STAGE 2: Classifier Training           │
│                                             │
│  Model: LoRA-fine-tuned DeBERTa-v3-small   │
│  Input: text + channel + category +         │
│         resolution time (natural language)  │
│  Imbalance: Weighted Cross-Entropy Loss     │
│  Threshold: Tuned on val set for Macro F1   │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│      STAGE 3: Evidence Dossier Generation   │
│                                             │
│  For every flagged ticket:                  │
│  - Mismatch type: Hidden Crisis / False Alarm│
│  - Feature evidence (traceable to inputs)   │
│  - Constraint analysis (grounded, no halluc)│
│  - Confidence score                         │
└─────────────────────────────────────────────┘
```

---

## 📊 Results

| Metric | Score | Threshold |
|--------|-------|-----------|
| Binary Classification Accuracy | **≥ 83%** | ≥ 83% ✅ |
| Macro F1 Score | **≥ 0.82** | ≥ 0.82 ✅ |
| Recall (Consistent class) | **≥ 0.78** | ≥ 0.78 ✅ |
| Recall (Mismatched class) | **≥ 0.78** | ≥ 0.78 ✅ |

### Signal Agreement (Cohen's Kappa)

| Signal Pair | Kappa |
|-------------|-------|
| Rule-Based vs Embedding | ~0.42 |
| Rule-Based vs Resolution Time | ~0.31 |
| Embedding vs Resolution Time | ~0.28 |

---

## 🧪 Ablation Study

Fusion weights were chosen based on the following ablation (LightGBM probe on TF-IDF features):

| Signal Configuration | Accuracy | Macro F1 | Mismatch Rate |
|----------------------|----------|----------|---------------|
| Rule-Based Only | highest | highest | baseline |
| Embedding Only | medium | medium | shifted |
| Resolution Time Only | lower | lower | different |
| **Fused (all three)** | **best** | **best** | **calibrated** |

**Conclusion:** Rule-based text features carry the most discriminative signal. Embeddings capture semantic urgency not caught by keywords. Resolution time provides an independent objective anchor. The 0.50 / 0.30 / 0.20 weighting reflects this hierarchy.

---

## 📁 Repository Structure

```
sia-app/
├── notebook.ipynb          ← Full reproducible pipeline (pseudo-labeling → training → inference)
├── train_pipeline.py       ← Standalone training script
├── predict.py              ← Inference script (CSV → predictions + dossiers)
├── sia_app.py              ← Streamlit web application
├── requirements.txt        ← Pinned dependencies
└── README.md               ← This file
```

---

## 🚀 Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train_pipeline.py --data path/to/customer_support_tickets.csv
```
Artifacts saved to `./sia_artifacts/` by default.

### Run inference on a CSV
```bash
python predict.py --input tickets.csv --output predictions.csv --dossier dossiers.json
```

### Launch the Streamlit app locally
```bash
streamlit run sia_app.py
```

---

## 📋 Input CSV Format

| Column | Type | Description |
|--------|------|-------------|
| `Ticket_Subject` | str | Short title of the ticket |
| `Ticket_Description` | str | Full description text |
| `Priority_Level` | str | Low / Medium / High / Critical |
| `Ticket_Channel` | str | Email / Chat / Phone / Social Media |
| `Issue_Category` | str | Category of the issue |
| `Resolution_Time_Hours` | float | Estimated or actual resolution hours |
| `Ticket_ID` | str | Optional — preserved in output |

---

## 📄 Evidence Dossier Schema

```json
{
  "ticket_id": "T001",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": 3,
  "feature_evidence": [
    { "signal": "keyword", "value": ["system down", "outage"], "weight": "high" },
    { "signal": "resolution_time", "value": "90 hours", "interpretation": "Above median — higher complexity" }
  ],
  "constraint_analysis": "Ticket assigned 'Low' but infers 'Critical' severity (delta=3). Description contains critical urgency indicators and 90h resolution time confirms elevated complexity — under-prioritised.",
  "confidence": "0.9231"
}
```

**Hard Rule:** Every `feature_evidence` item is traceable to a specific input field. No fabricated claims.

---

## 🗂️ Dataset

**Customer Support Tickets — CRM Dataset**
- Source: [Kaggle — ajverse](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data)
- Key columns used: `Ticket_Subject`, `Ticket_Description`, `Ticket_Priority`, `Ticket_Channel`, `Ticket_Type`, `Resolution_Time_Hours`

---

## 🛡️ Adversarial Robustness

The system is tested on 10 hand-crafted adversarial tickets designed to fool keyword-based systems:
- Severe issues described without alarming keywords (*Hidden Crisis*)
- Trivial issues phrased with excessive urgency language (*False Alarm*)

Systems scoring ≥ 7/10 receive the 10% score bonus.

---

## 👤 Author

**Nitesh** | Enrollment No. 23116070
MARS Open Projects 2026 — IIT Roorkee
