"""
predict.py — SIA Inference Script
Usage:
    python predict.py --input tickets.csv --output predictions.csv
"""

import os
import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Keywords (Stage 1 — Rule-Based NLP) ──────────────────────────
CRITICAL_KEYWORDS = [
    "data loss", "breach", "hacked", "corrupted", "not working",
    "system down", "cannot access", "urgent", "emergency", "critical",
    "lost all", "deleted", "security", "virus", "malware", "ransomware",
    "down", "outage", "failure", "crashed", "broken", "immediately"
]
HIGH_KEYWORDS = [
    "error", "failed", "issue", "problem", "not syncing", "slow",
    "freezing", "crash", "bug", "incorrect", "wrong", "missing",
    "unable", "cannot", "keeps failing", "not loading", "stuck"
]
MEDIUM_KEYWORDS = [
    "help", "question", "how to", "confused", "need assistance",
    "not sure", "wondering", "inquiry", "clarification", "update"
]
LOW_KEYWORDS = [
    "where", "when", "what time", "hours", "location", "information",
    "curious", "just checking", "fyi", "general", "feedback"
]
ESCALATION_PHRASES = [
    "escalate", "manager", "supervisor", "legal", "lawsuit",
    "refund", "compensation", "unacceptable", "terrible", "worst"
]
NEGATION_PATTERNS = [
    r"not\s+\w+", r"never\s+\w+", r"no\s+\w+", r"cannot\s+\w+",
    r"can't\s+\w+", r"won't\s+\w+", r"doesn't\s+\w+"
]

PRIORITY_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
SEV_MAP      = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
W_RULE, W_EMBEDDING, W_RESOLUTION = 0.50, 0.30, 0.20
ARTIFACT_DIR = "./sia_artifacts"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\.\!\?\,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rule_based_severity(text):
    t = str(text).lower()
    score = (
        sum(1 for kw in CRITICAL_KEYWORDS  if kw in t) * 4 +
        sum(1 for kw in HIGH_KEYWORDS       if kw in t) * 3 +
        sum(1 for kw in MEDIUM_KEYWORDS     if kw in t) * 2 +
        sum(1 for kw in LOW_KEYWORDS        if kw in t) * 1 +
        sum(1 for ep in ESCALATION_PHRASES  if ep in t) * 3 +
        sum(1 for p  in NEGATION_PATTERNS   if re.search(p, t)) * 2
    )
    if score >= 10: return 4
    elif score >= 6: return 3
    elif score >= 3: return 2
    else: return 1


def resolution_to_severity(hours, p25=24, p50=48, p75=72):
    if hours <= p25:   return 1
    elif hours <= p50: return 2
    elif hours <= p75: return 3
    else:              return 4


def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence transformer...")
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Warning: Could not load embedder ({e}). Using rule-only mode.")
        return None


def embedding_severity(text, embedder):
    if embedder is None:
        return 2
    references = {
        4: "system down critical failure emergency data breach cannot access outage",
        3: "error failed problem bug crash incorrect missing unable keeps failing",
        2: "help question how to confused need assistance inquiry clarification",
        1: "where when hours location information feedback general curious",
    }
    ref_texts = list(references.values())
    ref_sevs  = list(references.keys())
    text_emb  = embedder.encode([text])[0]
    ref_embs  = embedder.encode(ref_texts)
    sims = ref_embs @ text_emb / (np.linalg.norm(ref_embs, axis=1) * np.linalg.norm(text_emb) + 1e-9)
    return ref_sevs[int(np.argmax(sims))]


def load_deberta():
    model_path     = os.path.join(ARTIFACT_DIR, "deberta_lora_adapter")
    tokenizer_path = os.path.join(ARTIFACT_DIR, "tokenizer")
    threshold_path = os.path.join(ARTIFACT_DIR, "best_threshold.pkl")
    if not os.path.exists(model_path):
        print("DeBERTa artifacts not found. Using Stage 1 only.")
        return None, None, 0.5
    try:
        import joblib, torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        tokenizer  = AutoTokenizer.from_pretrained(tokenizer_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-small", num_labels=2)
        model      = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        threshold  = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.5
        print("DeBERTa model loaded successfully.")
        return model, tokenizer, float(threshold)
    except Exception as e:
        print(f"Warning: Could not load DeBERTa ({e}).")
        return None, None, 0.5


def generate_dossier(row, mismatch_type, inferred_label, delta, confidence):
    t = str(row.get("clean_text", "")).lower()
    found_critical = [kw for kw in CRITICAL_KEYWORDS if kw in t]
    found_high     = [kw for kw in HIGH_KEYWORDS     if kw in t]
    found_escalate = [kw for kw in ESCALATION_PHRASES if kw in t]

    feature_evidence = []
    if found_critical:
        feature_evidence.append({"signal": "keyword", "value": found_critical[:3], "weight": "high"})
    if found_high:
        feature_evidence.append({"signal": "keyword", "value": found_high[:3], "weight": "medium"})
    if found_escalate:
        feature_evidence.append({"signal": "keyword", "value": found_escalate[:2], "weight": "high"})

    res_time = float(row.get("Resolution_Time_Hours", 48))
    feature_evidence.append({
        "signal": "resolution_time",
        "value": f"{res_time:.0f} hours",
        "interpretation": "Above median — higher complexity" if res_time > 48 else "Below median — simpler issue"
    })
    feature_evidence.append({"signal": "ticket_channel",  "value": str(row.get("Ticket_Channel", "")),  "weight": "low"})
    feature_evidence.append({"signal": "issue_category",  "value": str(row.get("Issue_Category", "")),  "weight": "high" if row.get("Issue_Category") in ["Fraud", "Technical"] else "low"})

    assigned = row.get("Priority_Level", "Medium")
    if mismatch_type == "Hidden Crisis":
        analysis = (f"Ticket was assigned '{assigned}' priority but text analysis infers '{inferred_label}' severity "
                    f"(delta={delta}). Urgency indicators detected in description suggest under-prioritisation.")
    else:
        analysis = (f"Ticket was assigned '{assigned}' priority but text analysis infers only '{inferred_label}' severity "
                    f"(delta={delta}). Lack of critical indicators suggests over-prioritisation.")

    return {
        "ticket_id":         str(row.get("Ticket_ID", "N/A")),
        "assigned_priority": assigned,
        "inferred_severity": inferred_label,
        "mismatch_type":     mismatch_type,
        "severity_delta":    int(delta),
        "feature_evidence":  feature_evidence,
        "constraint_analysis": analysis,
        "confidence":        f"{confidence:.4f}"
    }


def predict(input_csv, output_csv, dossier_json):
    print(f"\n{'='*55}")
    print("  SUPPORT INTEGRITY AUDITOR (SIA) — Inference")
    print(f"{'='*55}\n")

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} tickets from '{input_csv}'")

    required = {"Ticket_Subject", "Ticket_Description", "Priority_Level",
                "Ticket_Channel", "Issue_Category", "Resolution_Time_Hours"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    embedder                       = load_embedder()
    deberta_model, tokenizer, thr  = load_deberta()

    df["clean_text"] = (df["Ticket_Subject"].fillna("") + " [SEP] " + df["Ticket_Description"].fillna("")).apply(clean_text)
    df["priority_numeric"] = df["Priority_Level"].map(PRIORITY_MAP).fillna(2).astype(int)
    df["resolution_time_numeric"] = pd.to_numeric(df["Resolution_Time_Hours"], errors="coerce").fillna(48)

    print("\nRunning inference...")
    results, dossiers = [], []

    for idx, row in df.iterrows():
        rule_sev   = rule_based_severity(row["clean_text"])
        embed_sev  = embedding_severity(row["clean_text"], embedder)
        res_sev    = resolution_to_severity(row["resolution_time_numeric"])
        fused      = W_RULE * rule_sev + W_EMBEDDING * embed_sev + W_RESOLUTION * res_sev
        inferred   = max(1, min(4, round(fused)))
        delta      = abs(inferred - row["priority_numeric"])
        mismatch   = int(delta >= 2)
        confidence = fused / 4.0  # normalised to [0,1]

        # DeBERTa override
        if deberta_model is not None:
            import torch
            model_input = (
                f"Subject: {row['Ticket_Subject']} Description: {row['Ticket_Description']} "
                f"Channel: {row['Ticket_Channel']} Category: {row['Issue_Category']} "
                f"Assigned Priority: {row['Priority_Level']} Resolution Hours: {int(row['resolution_time_numeric'])}"
            )
            enc = tokenizer(model_input, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
            with torch.no_grad():
                logits = deberta_model(**enc).logits
            proba = torch.softmax(logits, dim=1)[0, 1].item()
            mismatch   = int(proba >= thr)
            confidence = proba

        if mismatch == 0:
            mtype = "Consistent"
        elif inferred > row["priority_numeric"]:
            mtype = "Hidden Crisis"
        else:
            mtype = "False Alarm"

        inferred_label = SEV_MAP[inferred]
        results.append({
            "SIA_Rule_Severity":     SEV_MAP.get(rule_sev, ""),
            "SIA_Embed_Severity":    SEV_MAP.get(embed_sev, ""),
            "SIA_Res_Severity":      SEV_MAP.get(res_sev, ""),
            "SIA_Inferred_Severity": inferred_label,
            "SIA_Severity_Delta":    delta,
            "SIA_Mismatch_Label":    mismatch,
            "SIA_Mismatch_Type":     mtype,
            "SIA_Confidence":        round(confidence, 4),
        })

        if mismatch == 1:
            d = generate_dossier(row, mtype, inferred_label, delta, confidence)
            dossiers.append(d)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(df)} tickets...")

    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    result_df.to_csv(output_csv, index=False)

    with open(dossier_json, "w") as f:
        json.dump(dossiers, f, indent=2)

    # Summary
    total      = len(result_df)
    mismatched = sum(r["SIA_Mismatch_Label"] for r in results)
    hidden     = sum(1 for r in results if r["SIA_Mismatch_Type"] == "Hidden Crisis")
    false_alrm = sum(1 for r in results if r["SIA_Mismatch_Type"] == "False Alarm")

    print(f"\n{'='*55}")
    print("  RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  Total tickets   : {total}")
    print(f"  Mismatched      : {mismatched} ({mismatched/total*100:.1f}%)")
    print(f"  Hidden Crisis   : {hidden}")
    print(f"  False Alarm     : {false_alrm}")
    print(f"  Consistent      : {total - mismatched}")
    print(f"\n  Output CSV      : {output_csv}")
    print(f"  Dossiers JSON   : {dossier_json}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA — Support Integrity Auditor Inference")
    parser.add_argument("--input",   required=True,                    help="Input CSV file path")
    parser.add_argument("--output",  default="predictions.csv",        help="Output CSV file path")
    parser.add_argument("--dossier", default="dossiers.json",          help="Output dossier JSON path")
    args = parser.parse_args()
    predict(args.input, args.output, args.dossier)
