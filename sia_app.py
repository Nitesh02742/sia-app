"""
Support Integrity Auditor (SIA) — Streamlit Web App
MARS Open Projects 2026 — Problem Statement 1

Modes:
  1. Single Ticket Test  — paste one ticket and get instant analysis
  2. Batch CSV Test      — upload a CSV of tickets and download results
"""

import os
import re
import json
import warnings
import time
import io

import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Support Integrity Auditor (SIA)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #4361ee;
        margin-bottom: 1rem;
    }
    .badge-consistent  { background:#d4edda; color:#155724; border-radius:8px; padding:3px 10px; font-weight:600; }
    .badge-hidden      { background:#f8d7da; color:#721c24; border-radius:8px; padding:3px 10px; font-weight:600; }
    .badge-false-alarm { background:#fff3cd; color:#856404; border-radius:8px; padding:3px 10px; font-weight:600; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .stAlert { border-radius: 10px; }
    .dossier-box {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #b0c4ff;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants / Keywords (from Cell 5)
# ─────────────────────────────────────────────
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
PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"]
CHANNEL_OPTIONS  = ["Email", "Chat", "Phone", "Social Media", "Other"]
CATEGORY_OPTIONS = [
    "Technical", "Billing", "General Inquiry", "Fraud",
    "Account", "Shipping", "Returns", "Other"
]

ARTIFACT_DIR   = "./sia_artifacts"
W_RULE         = 0.50
W_EMBEDDING    = 0.30
W_RESOLUTION   = 0.20

# ─────────────────────────────────────────────
# Core Functions (self-contained — no trained model needed for MVP)
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\.\!\?\,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rule_based_severity(text: str) -> int:
    t = str(text).lower()
    critical_hits = sum(1 for kw in CRITICAL_KEYWORDS  if kw in t)
    high_hits     = sum(1 for kw in HIGH_KEYWORDS       if kw in t)
    medium_hits   = sum(1 for kw in MEDIUM_KEYWORDS     if kw in t)
    low_hits      = sum(1 for kw in LOW_KEYWORDS        if kw in t)
    escalation    = sum(1 for ep in ESCALATION_PHRASES  if ep in t)
    negations     = sum(1 for pat in NEGATION_PATTERNS  if re.search(pat, t))

    score = (critical_hits * 4 + high_hits * 3 + medium_hits * 2
             + low_hits * 1 + escalation * 3 + negations * 2)

    if score >= 10:  return 4
    elif score >= 6: return 3
    elif score >= 3: return 2
    else:            return 1


def resolution_to_severity(hours: float, p25: float, p50: float, p75: float) -> int:
    if hours <= p25:   return 1
    elif hours <= p50: return 2
    elif hours <= p75: return 3
    else:              return 4


@st.cache_resource(show_spinner="Loading sentence encoder…")
def load_sentence_transformer():
    """Load SentenceTransformer — cached across reruns."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"Could not load SentenceTransformer: {e}. Falling back to rule-only mode.")
        return None


@st.cache_resource(show_spinner="Loading DeBERTa classifier…")
def load_deberta_model():
    """Load fine-tuned LoRA DeBERTa model if saved artifacts exist."""
    model_path    = os.path.join(ARTIFACT_DIR, "deberta_lora_adapter")
    tokenizer_path = os.path.join(ARTIFACT_DIR, "tokenizer")
    threshold_path = os.path.join(ARTIFACT_DIR, "best_threshold.pkl")

    if not os.path.exists(model_path):
        return None, None, 0.5

    try:
        import joblib
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-small", num_labels=2
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()

        threshold = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.5
        return model, tokenizer, float(threshold)
    except Exception as e:
        st.warning(f"Could not load DeBERTa model: {e}")
        return None, None, 0.5


def embedding_severity_single(text: str, embedder) -> int:
    """Estimate severity via semantic similarity to reference sentences."""
    if embedder is None:
        return 2  # neutral fallback

    references = {
        4: "system down critical failure emergency data breach cannot access outage",
        3: "error failed problem bug crash incorrect missing unable keeps failing not loading",
        2: "help question how to confused need assistance inquiry clarification",
        1: "where when hours location information feedback general curious",
    }
    ref_texts   = list(references.values())
    ref_sevs    = list(references.keys())
    text_emb    = embedder.encode([text])[0]
    ref_embs    = embedder.encode(ref_texts)

    sims = ref_embs @ text_emb / (
        np.linalg.norm(ref_embs, axis=1) * np.linalg.norm(text_emb) + 1e-9
    )
    return ref_sevs[int(np.argmax(sims))]


def infer_mismatch(
    subject: str,
    description: str,
    priority: str,
    channel: str,
    category: str,
    resolution_hours: float,
    embedder,
    deberta_model=None,
    tokenizer=None,
    threshold: float = 0.5,
    median_resolution: float = 48.0,
) -> dict:
    """
    Full SIA pipeline for a single ticket.
    Returns a rich result dict.
    """
    combined   = f"{subject} [SEP] {description}"
    clean      = clean_text(combined)
    rule_sev   = rule_based_severity(clean)
    embed_sev  = embedding_severity_single(clean, embedder)

    # Resolution percentiles: we only have one ticket, so use fixed reference quartiles
    # (from dataset: approximate p25≈24h, p50≈48h, p75≈72h)
    p25, p50, p75 = 24.0, 48.0, 72.0
    res_sev = resolution_to_severity(resolution_hours, p25, p50, p75)

    fused_score    = W_RULE * rule_sev + W_EMBEDDING * embed_sev + W_RESOLUTION * res_sev
    inferred_sev   = int(round(fused_score))
    inferred_sev   = max(1, min(4, inferred_sev))
    priority_num   = PRIORITY_MAP.get(priority, 2)
    delta          = abs(inferred_sev - priority_num)
    mismatch_label = int(delta >= 2)

    # DeBERTa override if available
    deberta_proba  = None
    if deberta_model is not None and tokenizer is not None:
        model_input = (
            f"Subject: {subject} Description: {description} "
            f"Channel: {channel} Category: {category} "
            f"Assigned Priority: {priority} Resolution Hours: {int(resolution_hours)}"
        )
        enc = tokenizer(model_input, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
        with torch.no_grad():
            logits = deberta_model(**enc).logits
        proba = torch.softmax(logits, dim=1)[0, 1].item()
        deberta_proba  = proba
        mismatch_label = int(proba >= threshold)

    if mismatch_label == 0:
        mismatch_type = "Consistent"
    elif inferred_sev > priority_num:
        mismatch_type = "Hidden Crisis"
    else:
        mismatch_type = "False Alarm"

    # Evidence
    t = clean.lower()
    found_critical = [kw for kw in CRITICAL_KEYWORDS  if kw in t]
    found_high     = [kw for kw in HIGH_KEYWORDS       if kw in t]
    found_escalate = [kw for kw in ESCALATION_PHRASES  if kw in t]

    feature_evidence = []
    if found_critical:
        feature_evidence.append({"signal": "Critical Keywords",  "value": found_critical[:4], "weight": "🔴 High"})
    if found_high:
        feature_evidence.append({"signal": "High-Severity Keywords", "value": found_high[:4], "weight": "🟠 Medium"})
    if found_escalate:
        feature_evidence.append({"signal": "Escalation Phrases", "value": found_escalate[:3], "weight": "🔴 High"})
    feature_evidence.append({
        "signal": "Resolution Time",
        "value": f"{resolution_hours:.0f} hours",
        "weight": "🟡 Context",
        "note": "Above reference median (48h) — higher complexity" if resolution_hours > median_resolution
                else "Below reference median (48h) — simpler issue"
    })
    feature_evidence.append({"signal": "Channel", "value": channel, "weight": "⚪ Low"})
    feature_evidence.append({
        "signal": "Category",
        "value": category,
        "weight": "🔴 High" if category in ["Fraud", "Technical"] else "⚪ Low"
    })

    return {
        "rule_sev": rule_sev,
        "embed_sev": embed_sev,
        "res_sev": res_sev,
        "fused_score": fused_score,
        "inferred_severity_num": inferred_sev,
        "inferred_severity_label": SEV_MAP[inferred_sev],
        "assigned_priority": priority,
        "priority_num": priority_num,
        "delta": delta,
        "mismatch_label": mismatch_label,
        "mismatch_type": mismatch_type,
        "deberta_proba": deberta_proba,
        "feature_evidence": feature_evidence,
    }


def process_batch_df(df_in: pd.DataFrame, embedder, deberta_model, tokenizer, threshold: float) -> pd.DataFrame:
    """Run SIA on every row of a DataFrame. Returns enriched DataFrame."""
    required = {"Ticket_Subject", "Ticket_Description", "Priority_Level",
                "Ticket_Channel", "Issue_Category", "Resolution_Time_Hours"}
    missing  = required - set(df_in.columns)
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return pd.DataFrame()

    results = []
    prog    = st.progress(0, text="Analysing tickets…")
    n       = len(df_in)

    for i, row in df_in.iterrows():
        res_hrs = pd.to_numeric(row["Resolution_Time_Hours"], errors="coerce")
        if pd.isna(res_hrs):
            res_hrs = 48.0

        r = infer_mismatch(
            subject=str(row.get("Ticket_Subject", "")),
            description=str(row.get("Ticket_Description", "")),
            priority=str(row.get("Priority_Level", "Medium")),
            channel=str(row.get("Ticket_Channel", "Other")),
            category=str(row.get("Issue_Category", "Other")),
            resolution_hours=float(res_hrs),
            embedder=embedder,
            deberta_model=deberta_model,
            tokenizer=tokenizer,
            threshold=threshold,
        )
        out_row = row.to_dict()
        out_row.update({
            "SIA_Rule_Severity":      SEV_MAP.get(r["rule_sev"], ""),
            "SIA_Embed_Severity":     SEV_MAP.get(r["embed_sev"], ""),
            "SIA_Res_Severity":       SEV_MAP.get(r["res_sev"], ""),
            "SIA_Inferred_Severity":  r["inferred_severity_label"],
            "SIA_Severity_Delta":     r["delta"],
            "SIA_Mismatch_Label":     r["mismatch_label"],
            "SIA_Mismatch_Type":      r["mismatch_type"],
            "SIA_DeBERTa_Proba":      round(r["deberta_proba"], 4) if r["deberta_proba"] is not None else "N/A",
        })
        results.append(out_row)
        prog.progress((i + 1) / n, text=f"Analysed {i+1}/{n} tickets…")

    prog.empty()
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=72)
    st.markdown("## 🔍 SIA — Support Integrity Auditor")
    st.markdown("**MARS Open Projects 2026**")
    st.divider()

    mode = st.radio(
        "Select Mode",
        ["🎫 Single Ticket Test", "📂 Batch CSV Test"],
        index=0,
    )

    st.divider()
    st.markdown("### ⚙️ Model Info")
    st.info(
        "**Stage 1:** Rule-Based NLP + Sentence Embeddings + Resolution Time\n\n"
        "**Stage 2:** LoRA fine-tuned DeBERTa-v3-small *(if artifacts present)*\n\n"
        "**Stage 3:** Evidence Dossier Generation"
    )

    use_embedder = st.checkbox("Use Sentence Embeddings", value=True)
    use_deberta  = st.checkbox("Use DeBERTa (if saved)", value=True)
    st.divider()
    st.caption("Built with ❤️ using Streamlit")

# ─────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────
embedder = load_sentence_transformer() if use_embedder else None
if use_deberta:
    deberta_model, deberta_tokenizer, best_threshold = load_deberta_model()
else:
    deberta_model, deberta_tokenizer, best_threshold = None, None, 0.5

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🔍 Support Integrity Auditor (SIA)</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Self-supervised pipeline to detect priority mismatches in customer support tickets — '
    'Hidden Crises & False Alarms.</p>',
    unsafe_allow_html=True,
)

model_status_cols = st.columns(3)
with model_status_cols[0]:
    st.metric("Sentence Embedder", "✅ Loaded" if embedder else "⚠️ Disabled")
with model_status_cols[1]:
    st.metric("DeBERTa-LoRA", "✅ Loaded" if deberta_model else "⚠️ Not found")
with model_status_cols[2]:
    st.metric("Decision Threshold", f"{best_threshold:.2f}")

st.divider()

# ─────────────────────────────────────────────
# Mode 1: Single Ticket Test
# ─────────────────────────────────────────────
if mode == "🎫 Single Ticket Test":
    st.markdown("## 🎫 Single Ticket Analysis")
    st.markdown("Fill in the ticket details below and click **Analyse Ticket**.")

    with st.form("single_ticket_form"):
        c1, c2 = st.columns(2)

        with c1:
            subject = st.text_input(
                "Ticket Subject *",
                placeholder="e.g. System down — cannot access dashboard",
            )
            description = st.text_area(
                "Ticket Description *",
                height=140,
                placeholder="Describe the issue in detail…",
            )

        with c2:
            priority         = st.selectbox("Assigned Priority *", PRIORITY_OPTIONS, index=1)
            channel          = st.selectbox("Ticket Channel", CHANNEL_OPTIONS, index=0)
            category         = st.selectbox("Issue Category", CATEGORY_OPTIONS, index=0)
            resolution_hours = st.number_input(
                "Resolution Time (hours) *",
                min_value=0.0, max_value=1000.0, value=48.0, step=1.0,
                help="Estimated or actual resolution time in hours"
            )

        submitted = st.form_submit_button("🔎 Analyse Ticket", use_container_width=True)

    if submitted:
        if not subject.strip() or not description.strip():
            st.warning("Please fill in Subject and Description.")
        else:
            with st.spinner("Running SIA pipeline…"):
                result = infer_mismatch(
                    subject=subject,
                    description=description,
                    priority=priority,
                    channel=channel,
                    category=category,
                    resolution_hours=resolution_hours,
                    embedder=embedder,
                    deberta_model=deberta_model,
                    tokenizer=deberta_tokenizer,
                    threshold=best_threshold,
                )

            # ── Result Banner ──
            mtype = result["mismatch_type"]
            if mtype == "Consistent":
                st.success("✅  **CONSISTENT** — Priority matches inferred severity.")
                badge_html = '<span class="badge-consistent">✅ Consistent</span>'
            elif mtype == "Hidden Crisis":
                st.error("🚨  **HIDDEN CRISIS** — Ticket is under-prioritised! Real severity is higher.")
                badge_html = '<span class="badge-hidden">🚨 Hidden Crisis</span>'
            else:
                st.warning("⚠️  **FALSE ALARM** — Ticket may be over-prioritised.")
                badge_html = '<span class="badge-false-alarm">⚠️ False Alarm</span>'

            # ── Metrics Row ──
            st.markdown("### 📊 Prediction Summary")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Assigned Priority",   result["assigned_priority"])
            mc2.metric("Inferred Severity",   result["inferred_severity_label"])
            mc3.metric("Severity Delta",       result["delta"])
            mc4.metric("Mismatch",             "Yes" if result["mismatch_label"] else "No")
            if result["deberta_proba"] is not None:
                mc5.metric("DeBERTa Confidence", f"{result['deberta_proba']:.3f}")
            else:
                mc5.metric("Rule Fused Score", f"{result['fused_score']:.2f}")

            # ── Signal Breakdown ──
            st.markdown("### 🧪 Signal Breakdown")
            sig_df = pd.DataFrame({
                "Signal":   ["Rule-Based NLP (50%)", "Sentence Embedding (30%)", "Resolution Time (20%)", "Fused Score"],
                "Severity": [
                    SEV_MAP[result["rule_sev"]],
                    SEV_MAP[result["embed_sev"]],
                    SEV_MAP[result["res_sev"]],
                    SEV_MAP[max(1, min(4, round(result["fused_score"])))],
                ],
                "Numeric":  [result["rule_sev"], result["embed_sev"], result["res_sev"],
                             round(result["fused_score"], 2)],
            })
            st.dataframe(sig_df, use_container_width=True, hide_index=True)

            # ── Evidence Dossier ──
            if result["mismatch_label"] == 1:
                st.markdown("### 📋 Evidence Dossier")
                st.markdown('<div class="dossier-box">', unsafe_allow_html=True)

                ev_data = []
                for ev in result["feature_evidence"]:
                    val = ev["value"] if isinstance(ev["value"], str) else ", ".join(ev["value"])
                    note = ev.get("note", "")
                    ev_data.append({"Signal": ev["signal"], "Evidence": val, "Weight": ev["weight"], "Note": note})

                ev_df = pd.DataFrame(ev_data)
                st.dataframe(ev_df, use_container_width=True, hide_index=True)

                # Constraint Analysis
                inferred_lbl = result["inferred_severity_label"]
                delta        = result["delta"]
                if mtype == "Hidden Crisis":
                    analysis = (
                        f"Ticket was assigned **{priority}** priority but the SIA pipeline infers "
                        f"**{inferred_lbl}** severity (delta = {delta} level(s)). "
                        f"The description contains urgency indicators and its resolution time of "
                        f"{resolution_hours:.0f}h supports elevated complexity — suggesting this ticket "
                        f"was **under-prioritised**."
                    )
                else:
                    analysis = (
                        f"Ticket was assigned **{priority}** priority but the SIA pipeline infers "
                        f"only **{inferred_lbl}** severity (delta = {delta} level(s)). "
                        f"The description lacks critical urgency indicators and its resolution time of "
                        f"{resolution_hours:.0f}h is consistent with a lower-severity issue — suggesting "
                        f"this ticket was **over-prioritised**."
                    )

                st.markdown(f"**Constraint Analysis:** {analysis}")
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Mini bar chart ──
            st.markdown("### 📈 Severity Signal Chart")
            fig, ax = plt.subplots(figsize=(7, 2.5))
            labels  = ["Rule NLP", "Embedding", "Res. Time", "Assigned\nPriority"]
            values  = [result["rule_sev"], result["embed_sev"],
                       result["res_sev"], result["priority_num"]]
            colors  = ["#4361ee", "#7209b7", "#f72585", "#4cc9f0"]
            bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.5)
            ax.set_xlim(0, 4.5)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(["Low", "Medium", "High", "Critical"])
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                        SEV_MAP.get(int(round(val)), str(val)), va="center", fontsize=9)
            ax.set_title("Severity Scores per Signal", fontsize=11, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

# ─────────────────────────────────────────────
# Mode 2: Batch CSV Test
# ─────────────────────────────────────────────
else:
    st.markdown("## 📂 Batch CSV Analysis")

    st.markdown("""
    Upload a CSV file with the following **required columns**:

    | Column | Description |
    |---|---|
    | `Ticket_Subject` | Short title of the ticket |
    | `Ticket_Description` | Full description text |
    | `Priority_Level` | Low / Medium / High / Critical |
    | `Ticket_Channel` | Email / Chat / Phone / Social Media |
    | `Issue_Category` | Category of the issue |
    | `Resolution_Time_Hours` | Numeric — estimated or actual hours |

    Optional: `Ticket_ID` and any other columns will be preserved.
    """)

    # ── Download sample CSV ──
    sample_data = {
        "Ticket_ID":           ["T001", "T002", "T003", "T004"],
        "Ticket_Subject":      [
            "System completely down",
            "Quick question about hours",
            "URGENT payment issue",
            "Minor UI feedback",
        ],
        "Ticket_Description":  [
            "Our entire production system has crashed. All customers cannot access the platform. Data may be lost.",
            "Just wondering what your office hours are for next week. No rush!",
            "This is URGENT!! Just wanted to say thanks for the quick reply, all is good now.",
            "The button color on the dashboard is slightly off. Low priority but wanted to mention it.",
        ],
        "Priority_Level":      ["Low", "Critical", "Critical", "High"],
        "Ticket_Channel":      ["Email", "Chat", "Phone", "Email"],
        "Issue_Category":      ["Technical", "General Inquiry", "Billing", "General Inquiry"],
        "Resolution_Time_Hours": [90, 1, 2, 4],
    }
    sample_df  = pd.DataFrame(sample_data)
    sample_csv = sample_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Sample CSV",
        data=sample_csv,
        file_name="sia_sample_tickets.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload your ticket CSV", type=["csv"])

    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(df_input)} tickets** — preview:")
        st.dataframe(df_input.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Analysis", use_container_width=True):
            df_results = process_batch_df(
                df_in=df_input,
                embedder=embedder,
                deberta_model=deberta_model,
                tokenizer=deberta_tokenizer,
                threshold=best_threshold,
            )

            if not df_results.empty:
                st.success(f"✅ Analysis complete — {len(df_results)} tickets processed.")

                # ── Summary Stats ──
                st.markdown("### 📊 Batch Summary")
                total      = len(df_results)
                mismatched = df_results["SIA_Mismatch_Label"].sum()
                hidden     = (df_results["SIA_Mismatch_Type"] == "Hidden Crisis").sum()
                false_alrm = (df_results["SIA_Mismatch_Type"] == "False Alarm").sum()
                consistent = (df_results["SIA_Mismatch_Type"] == "Consistent").sum()

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Total Tickets",   total)
                sc2.metric("🚨 Hidden Crisis", hidden)
                sc3.metric("⚠️ False Alarm",  false_alrm)
                sc4.metric("✅ Consistent",   consistent)

                # ── Charts ──
                fcol1, fcol2 = st.columns(2)

                with fcol1:
                    st.markdown("#### Mismatch Type Distribution")
                    fig1, ax1 = plt.subplots(figsize=(4.5, 3))
                    type_counts = df_results["SIA_Mismatch_Type"].value_counts()
                    colors_map  = {"Consistent": "#4CAF50", "Hidden Crisis": "#F44336", "False Alarm": "#FF9800"}
                    colors_list = [colors_map.get(t, "#999") for t in type_counts.index]
                    ax1.bar(type_counts.index, type_counts.values, color=colors_list, edgecolor="white")
                    ax1.set_ylabel("Count")
                    ax1.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig1)

                with fcol2:
                    st.markdown("#### Severity Delta Distribution")
                    fig2, ax2 = plt.subplots(figsize=(4.5, 3))
                    df_results["SIA_Severity_Delta"].hist(ax=ax2, bins=5, color="#4361ee", edgecolor="white")
                    ax2.set_xlabel("Severity Delta")
                    ax2.set_ylabel("Count")
                    ax2.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig2)

                # ── Full Results Table ──
                st.markdown("### 📋 Full Results")
                display_cols = (
                    [c for c in ["Ticket_ID", "Ticket_Subject", "Priority_Level"] if c in df_results.columns]
                    + ["SIA_Inferred_Severity", "SIA_Severity_Delta",
                       "SIA_Mismatch_Label", "SIA_Mismatch_Type", "SIA_DeBERTa_Proba"]
                )
                st.dataframe(df_results[display_cols], use_container_width=True)

                # ── Flagged Tickets ──
                flagged = df_results[df_results["SIA_Mismatch_Label"] == 1]
                if not flagged.empty:
                    st.markdown(f"### 🚨 Flagged Tickets ({len(flagged)})")
                    st.dataframe(flagged[display_cols], use_container_width=True)

                # ── Download Results ──
                csv_out = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Full Results CSV",
                    data=csv_out,
                    file_name="sia_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # ── Dossier JSON ──
                dossiers = []
                for _, row in flagged.iterrows():
                    dossiers.append({
                        "ticket_id":          str(row.get("Ticket_ID", "N/A")),
                        "assigned_priority":  row.get("Priority_Level", ""),
                        "inferred_severity":  row.get("SIA_Inferred_Severity", ""),
                        "mismatch_type":      row.get("SIA_Mismatch_Type", ""),
                        "severity_delta":     int(row.get("SIA_Severity_Delta", 0)),
                        "deberta_confidence": str(row.get("SIA_DeBERTa_Proba", "N/A")),
                    })

                json_out = json.dumps(dossiers, indent=2).encode("utf-8")
                st.download_button(
                    "⬇️ Download Evidence Dossiers JSON",
                    data=json_out,
                    file_name="sia_dossiers.json",
                    mime="application/json",
                    use_container_width=True,
                )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "SIA — Support Integrity Auditor | MARS Open Projects 2026 | "
    "LoRA DeBERTa-v3-small + Sentence Transformers + Rule-Based NLP"
)
