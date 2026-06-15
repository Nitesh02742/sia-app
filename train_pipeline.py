"""
train_pipeline.py — SIA Standalone Training Script
MARS Open Projects 2026 — Problem Statement 1

Usage:
    python train_pipeline.py --data path/to/customer_support_tickets.csv
"""

import os
import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix, cohen_kappa_score, roc_curve, auc
)
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import joblib

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Keywords ──────────────────────────────────────────────────────
CRITICAL_KEYWORDS = [
    "data loss","breach","hacked","corrupted","not working","system down",
    "cannot access","urgent","emergency","critical","lost all","deleted",
    "security","virus","malware","ransomware","down","outage","failure",
    "crashed","broken","immediately"
]
HIGH_KEYWORDS = [
    "error","failed","issue","problem","not syncing","slow","freezing",
    "crash","bug","incorrect","wrong","missing","unable","cannot",
    "keeps failing","not loading","stuck"
]
MEDIUM_KEYWORDS = [
    "help","question","how to","confused","need assistance","not sure",
    "wondering","inquiry","clarification","update"
]
LOW_KEYWORDS = [
    "where","when","what time","hours","location","information",
    "curious","just checking","fyi","general","feedback"
]
ESCALATION_PHRASES = [
    "escalate","manager","supervisor","legal","lawsuit",
    "refund","compensation","unacceptable","terrible","worst"
]
NEGATION_PATTERNS = [
    r"not\s+\w+",r"never\s+\w+",r"no\s+\w+",r"cannot\s+\w+",
    r"can't\s+\w+",r"won't\s+\w+",r"doesn't\s+\w+"
]

PRIORITY_MAP = {"Low":1,"Medium":2,"High":3,"Critical":4}
SEV_MAP      = {1:"Low",2:"Medium",3:"High",4:"Critical"}
W_RULE, W_EMBEDDING, W_RESOLUTION = 0.50, 0.30, 0.20
MODEL_NAME   = "microsoft/deberta-v3-small"
MAX_LEN      = 256


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


def resolution_to_severity(hours, p25, p50, p75):
    if hours <= p25:   return 1
    elif hours <= p50: return 2
    elif hours <= p75: return 3
    else:              return 4


def build_input_text(row):
    return (
        f"Subject: {row['Ticket_Subject']} "
        f"Description: {row['Ticket_Description']} "
        f"Channel: {row['Ticket_Channel']} "
        f"Category: {row['Issue_Category']} "
        f"Assigned Priority: {row['Priority_Level']} "
        f"Resolution Hours: {int(row['resolution_time_numeric'])}"
    )


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train(data_path, artifact_dir, epochs=4):
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print("  SIA — Training Pipeline")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # ── Load Data ────────────────────────────────────────────────
    print("Step 1: Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Dataset shape: {df.shape}")

    df = df.dropna(subset=["Ticket_Subject","Ticket_Description","Priority_Level"]).reset_index(drop=True)
    df["Priority_Level"] = df["Priority_Level"].str.strip().str.title()
    df["combined_text"]  = df["Ticket_Subject"].fillna("") + " [SEP] " + df["Ticket_Description"].fillna("")
    df["clean_text"]     = df["combined_text"].apply(clean_text)
    df["priority_numeric"] = df["Priority_Level"].map(PRIORITY_MAP).fillna(2).astype(int)
    df["resolution_time_numeric"] = pd.to_numeric(df["Resolution_Time_Hours"], errors="coerce")
    df["resolution_time_numeric"] = df["resolution_time_numeric"].fillna(df["resolution_time_numeric"].median())

    channel_encoder  = LabelEncoder()
    category_encoder = LabelEncoder()
    df["channel_encoded"]  = channel_encoder.fit_transform(df["Ticket_Channel"].fillna("Unknown"))
    df["category_encoded"] = category_encoder.fit_transform(df["Issue_Category"].fillna("Unknown"))

    # ── Stage 1 Signal 1: Rule-Based NLP ─────────────────────────
    print("\nStep 2: Generating pseudo-labels...")
    print("  Signal 1 — Rule-based NLP severity scoring...")
    df["rule_based_severity"] = df["clean_text"].apply(rule_based_severity)

    # ── Stage 1 Signal 2: Sentence Embedding Clustering ──────────
    print("  Signal 2 — Sentence embedding clustering...")
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(df["combined_text"].tolist(), batch_size=64,
                                  show_progress_bar=True, convert_to_numpy=True)
    np.save(os.path.join(artifact_dir, "embeddings.npy"), embeddings)

    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    df["cluster"] = kmeans.fit_predict(embeddings)
    cluster_map   = (df.groupby("cluster")["rule_based_severity"].mean()
                       .rank(method="dense").astype(int))
    df["embedding_severity"] = df["cluster"].map(cluster_map)

    # ── Stage 1 Signal 3: Resolution Time ────────────────────────
    print("  Signal 3 — Resolution time bucketing...")
    p25 = df["resolution_time_numeric"].quantile(0.25)
    p50 = df["resolution_time_numeric"].quantile(0.50)
    p75 = df["resolution_time_numeric"].quantile(0.75)
    df["resolution_severity"] = df["resolution_time_numeric"].apply(
        lambda h: resolution_to_severity(h, p25, p50, p75)
    )

    # ── Fusion ────────────────────────────────────────────────────
    df["fused_severity_score"] = (
        W_RULE       * df["rule_based_severity"]
        + W_EMBEDDING  * df["embedding_severity"]
        + W_RESOLUTION * df["resolution_severity"]
    )
    df["inferred_severity"] = df["fused_severity_score"].round().astype(int).clip(1,4)
    df["severity_delta"]    = (df["inferred_severity"] - df["priority_numeric"]).abs()
    df["mismatch_label"]    = (df["severity_delta"] >= 2).astype(int)

    kappa_re = cohen_kappa_score(df["rule_based_severity"], df["embedding_severity"])
    kappa_rr = cohen_kappa_score(df["rule_based_severity"], df["resolution_severity"])
    kappa_er = cohen_kappa_score(df["embedding_severity"],  df["resolution_severity"])

    print(f"\n  Signal Agreement (Cohen's Kappa):")
    print(f"    Rule vs Embedding  : {kappa_re:.4f}")
    print(f"    Rule vs Resolution : {kappa_rr:.4f}")
    print(f"    Embed vs Resolution: {kappa_er:.4f}")
    print(f"  Mismatch rate: {df['mismatch_label'].mean()*100:.1f}%")

    # ── Stage 2: DeBERTa LoRA ────────────────────────────────────
    print("\nStep 3: Stage 2 — LoRA fine-tuning DeBERTa-v3-small...")
    df["model_input"] = df.apply(build_input_text, axis=1)

    train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["mismatch_label"])
    train_df, val_df      = train_test_split(train_val_df, test_size=0.20, random_state=SEED, stratify=train_val_df["mismatch_label"])
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(examples["model_input"], padding="max_length", truncation=True, max_length=MAX_LEN)

    def make_dataset(df_):
        ds = Dataset.from_dict({"model_input": df_["model_input"].tolist(), "label": df_["mismatch_label"].tolist()})
        return ds.map(tokenize, batched=True, remove_columns=["model_input"])

    train_tok, val_tok, test_tok = make_dataset(train_df), make_dataset(val_df), make_dataset(test_df)

    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    lora_cfg   = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                             target_modules=["query_proj","value_proj"],
                             lora_dropout=0.1, bias="none")
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    n0, n1        = (train_df["mismatch_label"]==0).sum(), (train_df["mismatch_label"]==1).sum()
    class_weights = torch.tensor([1.0, n0/max(n1,1)], dtype=torch.float)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {
            "accuracy":          accuracy_score(labels, preds),
            "macro_f1":          f1_score(labels, preds, average="macro"),
            "recall_consistent": recall_score(labels, preds, pos_label=0),
            "recall_mismatch":   recall_score(labels, preds, pos_label=1),
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(artifact_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # ── Threshold Tuning ─────────────────────────────────────────
    print("\nStep 4: Threshold tuning on validation set...")
    val_logits = trainer.predict(val_tok).predictions
    val_proba  = torch.softmax(torch.tensor(val_logits), dim=1)[:,1].numpy()
    y_val      = val_df["mismatch_label"].values

    best_threshold, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.2, 0.8, 0.01):
        preds   = (val_proba >= thresh).astype(int)
        f1      = f1_score(y_val, preds, average="macro")
        recalls = recall_score(y_val, preds, average=None)
        if f1 > best_f1 and recalls[0] >= 0.78 and recalls[1] >= 0.78:
            best_f1, best_threshold = f1, thresh

    print(f"  Best threshold: {best_threshold:.2f} | Val Macro F1: {best_f1:.4f}")

    # ── Test Evaluation ───────────────────────────────────────────
    print("\nStep 5: Final test set evaluation...")
    test_logits = trainer.predict(test_tok).predictions
    test_proba  = torch.softmax(torch.tensor(test_logits), dim=1)[:,1].numpy()
    y_test      = test_df["mismatch_label"].values
    test_preds  = (test_proba >= best_threshold).astype(int)

    test_acc      = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")
    test_recall   = recall_score(y_test, test_preds, average=None)

    print(f"\n{'='*60}")
    print("  FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy           : {test_acc*100:.2f}%  (threshold >= 83%)")
    print(f"  Macro F1           : {test_macro_f1:.4f}  (threshold >= 0.82)")
    print(f"  Recall (Consistent): {test_recall[0]:.4f}  (threshold >= 0.78)")
    print(f"  Recall (Mismatched): {test_recall[1]:.4f}  (threshold >= 0.78)")
    print(f"\n{classification_report(y_test, test_preds, target_names=['Consistent','Mismatched'])}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Consistent","Mismatched"],
                yticklabels=["Consistent","Mismatched"])
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "confusion_matrix.png"), dpi=120)
    print(f"  Confusion matrix saved.")

    # ── Save All Artifacts ────────────────────────────────────────
    print("\nStep 6: Saving artifacts...")
    model.save_pretrained(os.path.join(artifact_dir, "deberta_lora_adapter"))
    tokenizer.save_pretrained(os.path.join(artifact_dir, "tokenizer"))
    joblib.dump(channel_encoder,  os.path.join(artifact_dir, "channel_encoder.pkl"))
    joblib.dump(category_encoder, os.path.join(artifact_dir, "category_encoder.pkl"))
    joblib.dump(best_threshold,   os.path.join(artifact_dir, "best_threshold.pkl"))

    metrics = {
        "test_accuracy":           float(test_acc),
        "test_macro_f1":           float(test_macro_f1),
        "test_recall_consistent":  float(test_recall[0]),
        "test_recall_mismatched":  float(test_recall[1]),
        "best_threshold":          float(best_threshold),
        "signal_agreement_kappa": {
            "rule_vs_embedding":   float(kappa_re),
            "rule_vs_resolution":  float(kappa_rr),
            "embedding_vs_resolution": float(kappa_er),
        }
    }
    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  All artifacts saved to '{artifact_dir}/'")
    print(f"{'='*60}\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA — Training Pipeline")
    parser.add_argument("--data",     required=True,              help="Path to input CSV dataset")
    parser.add_argument("--artifacts",default="./sia_artifacts",  help="Directory to save model artifacts")
    parser.add_argument("--epochs",   default=4, type=int,        help="Number of training epochs")
    args = parser.parse_args()
    train(args.data, args.artifacts, args.epochs)
