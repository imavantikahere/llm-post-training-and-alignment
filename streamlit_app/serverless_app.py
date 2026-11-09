
"""LLM Post-Training Lab — benchmark dashboard + live Qwen GPU inference.

Run: streamlit run app.py
Streamlit runs on CPU; a remote RunPod GPU serves trained adapters.
"""

from __future__ import annotations

import re
import os
import json
import hashlib
import time
from pathlib import Path

import requests
import pandas as pd
import altair as alt
import streamlit as st
from serverless_client import run_serverless_job


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

STAGES = ["Base Qwen", "CPT", "SFT", "DPO"]
MODEL_COLORS = {"Base Qwen": "#4E8FBE", "CPT": "#83BDE6", "SFT": "#A68DDB", "DPO": "#E8A3C7"}


# Original seven fields used for the saved research benchmark.
REQUIRED_FIELDS = [
    "Equipment",
    "Reported issue",
    "Observed evidence",
    "Possible contributors (not confirmed)",
    "Missing information",
    "Recommended follow-up",
    "Assessment",
]

# Headings used by the live industrial maintenance report format.
# These are evaluated separately from the saved benchmark.
LIVE_REPORT_FIELDS = [
    "Equipment",
    "Observed symptoms",
    "Operating conditions",
    "Measurements",
    "Immediate actions",
    "Possible causes",
    "Recommended follow-up",
]

st.set_page_config(
    page_title="Industrial Maintenance Document Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: #C6E3F8 !important;
        color: #244B70 !important;
    }
    [data-testid="stHeader"] { background: rgba(198,227,248,.94) !important; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background: #B3D7F1 !important;
    }
    .block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1320px; }
    h1, h2, h3, h4, p, label, .stMarkdown { color: #244B70; }
    [data-testid="stMetric"] {
        background: #EAF6FF; border: 1px solid #A7D0EB;
        border-radius: 15px; padding: 14px 18px;
    }
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    [data-baseweb="select"] > div,
    [data-testid="stCodeBlock"] pre,
    [data-testid="stExpander"] details {
        background: #EAF6FF !important;
        color: #244B70 !important;
        border-color: #9FCBE8 !important;
        border-radius: 12px !important;
    }
    [data-testid="stBaseButton-primary"] {
        background: #407FB0 !important; color: white !important;
        border-radius: 12px !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #244B70 !important; border-bottom-color: #407FB0 !important;
    }
    .hero-panel {
        padding: 25px 30px;
        background: linear-gradient(115deg,#D7EDFD,#EAF6FF);
        border: 1px solid #A6CDEB; border-radius: 21px; margin-bottom: 20px;
    }
    .hero-eyebrow { color: #527FA3; font-size: .78rem; font-weight: 700; letter-spacing: .12em; }
    .hero-title { color: #173E63; font-weight: 750; font-size: 2.3rem; line-height: 1.17; margin: .55rem 0; }
    .hero-subtitle { color: #346B96; font-size: 1.12rem; font-weight: 600; }
    .soft-card {
        background: #EAF6FF; padding: 18px 20px;
        border: 1px solid #A7D0EB; border-radius: 15px;
        min-height: 135px; margin: 5px 0;
    }
    .soft-card p, .soft-card strong { color: #244B70; }
    .stage-flow {
        background: #EAF6FF; padding: 17px; border-radius: 14px;
        border: 1px solid #A7D0EB; color: #244B70;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. LOAD BENCHMARK DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence = pd.read_csv(DATA / "evidence_reviewed.csv")
    errors = pd.read_csv(DATA / "error_reviewed.csv")

    evidence_cols = {
        "id",
        "stage",
        "evidence",
        "verified_present",
        "technician_note",
        "model_response",
    }

    error_cols = {
        "id",
        "stage",
        "unsupported_observation",
        "unsupported_interpretation",
        "truncated",
        "model_response",
        "technician_note",
    }

    if missing := evidence_cols - set(evidence.columns):
        raise ValueError(
            f"Evidence CSV missing columns: {sorted(missing)}"
        )

    if missing := error_cols - set(errors.columns):
        raise ValueError(
            f"Error CSV missing columns: {sorted(missing)}"
        )

    if evidence.duplicated(["id", "stage", "evidence"]).any():
        raise ValueError("Duplicate evidence review rows")

    if errors.duplicated(["id", "stage"]).any():
        raise ValueError("Duplicate report review rows")

    if set(zip(evidence.id, evidence.stage)) != set(
        zip(errors.id, errors.stage)
    ):
        raise ValueError(
            "The evidence and error CSVs have different report identifiers"
        )

    for df, cols in [
        (evidence, ["verified_present"]),
        (
            errors,
            [
                "unsupported_observation",
                "unsupported_interpretation",
                "truncated",
            ],
        ),
    ]:
        for col in cols:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            if df[col].isna().any() or not df[col].isin([0, 1]).all():
                raise ValueError(
                    f"Column {col} must be completely reviewed as 0 or 1"
                )

            df[col] = df[col].astype(int)

    return evidence, errors


# ============================================================
# 3. BENCHMARK SCORING
# ============================================================

def count_fields(response: str) -> int:
    """
    Original benchmark heading-presence metric.

    This checks the original seven benchmark headings.
    It does not establish factual correctness.
    """
    return sum(
        bool(
            re.search(
                re.escape(name) + r"\s*:",
                response,
                re.I,
            )
        )
        for name in REQUIRED_FIELDS
    )


def compute_scores(
    evidence: pd.DataFrame,
    errors: pd.DataFrame,
):
    case_recalls = (
        evidence.groupby(
            ["id", "stage"],
            as_index=False,
        )["verified_present"]
        .mean()
        .rename(
            columns={
                "verified_present": "evidence_recall"
            }
        )
    )

    reports = errors.merge(
        case_recalls,
        on=["id", "stage"],
        validate="one_to_one",
    )

    reports["required_fields_present"] = (
        reports["model_response"].map(count_fields)
    )

    reports["field_coverage"] = (
        reports["required_fields_present"] / 7
    )

    reports["all_seven_fields"] = (
        reports["required_fields_present"] == 7
    ).astype(int)

    reports["any_unsupported_claim"] = (
        (reports["unsupported_observation"] > 0)
        | (reports["unsupported_interpretation"] > 0)
    ).astype(int)

    # Macro average: every scenario receives equal weight.
    stage = reports.groupby(
        "stage",
        as_index=False,
    ).agg(
        recall=("evidence_recall", "mean"),
        field_coverage=("field_coverage", "mean"),
        full_fields=("all_seven_fields", "sum"),
        unsupported_observations=(
            "unsupported_observation",
            "sum",
        ),
        unsupported_interpretations=(
            "unsupported_interpretation",
            "sum",
        ),
        truncated=("truncated", "sum"),
        cases=("id", "nunique"),
    )

    stage["stage"] = pd.Categorical(
        stage["stage"],
        categories=STAGES,
        ordered=True,
    )

    stage = stage.sort_values(
        "stage"
    ).reset_index(drop=True)

    stage["stage"] = stage["stage"].astype(str)

    return reports, stage


def show_report(text: str):
    st.code(
        text,
        language=None,
        wrap_lines=True,
    )


# ============================================================
# 4. NEW — LIVE INFERENCE METRICS
# ============================================================

def live_field_detected(response: str, field: str) -> bool:
    """Match live headings and explicitly allowed qualified variants."""
    if not isinstance(response, str):
        return False
    suffix = r"(?:\s*\(not confirmed\))?" if field == "Possible causes" else ""
    return bool(re.search(
        r"^\s*" + re.escape(field) + suffix + r"\s*:",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    ))


def count_live_report_fields(response: str) -> int:
    """Structural presence only, not an evidence-accuracy score."""
    return sum(live_field_detected(response, field) for field in LIVE_REPORT_FIELDS)

def safe_number(value):
    """Convert a numeric API value safely."""
    try:
        if value is None or isinstance(value, bool):
            return None

        result = float(value)

        if not pd.notna(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def build_live_metrics(outputs: dict) -> pd.DataFrame:
    """
    Convert real GPU responses into chart-ready metrics.

    Expected output fields:
        response
        generated_tokens
        latency_seconds
        reached_token_limit
    """
    rows = []

    for stage in STAGES:
        if stage not in outputs:
            continue

        item = outputs[stage]

        if not isinstance(item, dict):
            continue

        response = str(item.get("response") or "")

        tokens = safe_number(
            item.get("generated_tokens")
        )

        seconds = safe_number(
            item.get("latency_seconds")
        )

        tokens_per_second = None

        if (
            tokens is not None
            and seconds is not None
            and seconds > 0
        ):
            tokens_per_second = tokens / seconds

        detected_fields = count_live_report_fields(
            response
        )

        rows.append(
            {
                "Model": stage,
                "Latency (s)": (
                    round(seconds, 2)
                    if seconds is not None
                    else None
                ),
                "Generated tokens": (
                    int(tokens)
                    if tokens is not None
                    else None
                ),
                "Tokens/sec": (
                    round(tokens_per_second, 2)
                    if tokens_per_second is not None
                    else None
                ),
                "Live report fields": detected_fields,
                "Field coverage (%)": round(
                    detected_fields / 7 * 100,
                    1,
                ),
                "Reached token limit": bool(
                    item.get("reached_token_limit", False)
                ),
            }
        )

    return pd.DataFrame(rows)


def colored_live_bar_chart(df: pd.DataFrame, field: str, axis_title: str, maximum=None):
    """Fixed model order and colors across all live charts."""
    chart_data = df[["Model", field]].dropna()
    if chart_data.empty:
        st.info("No data returned for this metric.")
        return
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Model:N", sort=STAGES, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(field + ":Q", title=axis_title,
                    scale=alt.Scale(domain=[0, maximum]) if maximum else alt.Scale(zero=True)),
            color=alt.Color("Model:N",
                            scale=alt.Scale(domain=STAGES,
                                            range=[MODEL_COLORS[m] for m in STAGES]),
                            legend=None),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip(field + ":Q", format=".2f")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def show_live_comparison(outputs: dict):
    """
    Display charts after actual live inference.

    All values come from the RunPod response or
    deterministic heading detection.
    """

    df = build_live_metrics(outputs)

    if df.empty:
        st.info(
            "No live model results are available to compare."
        )
        return

    st.divider()

    st.header("Live Model Comparison")

    st.caption(
        "Metrics calculated from the current live GPU inference. "
        "These are not the saved benchmark results."
    )

    # -------------------------
    # Summary table
    # -------------------------

    st.subheader("Performance summary")

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )

    if len(df) == 1:
        st.info(
            "Only one checkpoint was generated. "
            "Select two or more checkpoints and generate again "
            "to compare model behavior."
        )

    # -------------------------
    # Interactive chart tabs
    # -------------------------

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Latency",
            "Generation speed",
            "Schema coverage",
            "Token usage",
        ]
    )

    # CHART 1: LATENCY

    with tab1:
        st.subheader(
            "Inference latency by checkpoint"
        )

        latency_df = (
            df.set_index("Model")[["Latency (s)"]]
            .dropna()
        )

        if not latency_df.empty:
            colored_live_bar_chart(df, "Latency (s)", "Seconds")
        else:
            st.info(
                "Latency measurements were not returned by the API."
            )

        st.caption(
            "Reported time per checkpoint. Depending on the "
            "backend timer, this may include model loading "
            "and other overhead. Sequential execution can "
            "affect comparisons."
        )

    # CHART 2: GENERATION SPEED

    with tab2:
        st.subheader(
            "Generation speed by checkpoint"
        )

        speed_df = (
            df.set_index("Model")[["Tokens/sec"]]
            .dropna()
        )

        if not speed_df.empty:
            colored_live_bar_chart(df, "Tokens/sec", "Tokens per second")
        else:
            st.info(
                "Generation speed could not be calculated."
            )

        st.caption(
            "Calculated as generated tokens divided by "
            "reported latency. If loading time is included, "
            "this is an end-to-end throughput proxy, not "
            "pure decoding throughput."
        )

    # CHART 3: LIVE SCHEMA PRESENCE

    with tab3:
        st.subheader(
            "Live report structure"
        )

        schema_df = df.set_index("Model")[
            ["Live report fields"]
        ]

        colored_live_bar_chart(df, "Live report fields", "Headings detected (out of 7)", maximum=7)

        st.caption(
            "Counts approved headings from the live seven-field "
            "maintenance-report format. This is a deterministic "
            "structural check, not factual accuracy."
        )

        with st.expander(
            "Which seven report fields are checked?"
        ):
            for field in LIVE_REPORT_FIELDS:
                st.write(f"• {field}")

        st.caption(
            "The historical benchmark uses a different "
            "seven-field heading specification and remains "
            "separate from this live structural check."
        )

    # CHART 4: GENERATED TOKENS

    with tab4:
        st.subheader(
            "Generated tokens by checkpoint"
        )

        tokens_df = (
            df.set_index("Model")[["Generated tokens"]]
            .dropna()
        )

        if not tokens_df.empty:
            colored_live_bar_chart(df, "Generated tokens", "Generated tokens")
        else:
            st.info(
                "Token counts were not returned by the API."
            )

        st.caption(
            "A larger response is not necessarily more accurate. "
            "Token count measures output length, not quality."
        )

    # -------------------------
    # Model-by-model heading detail
    # -------------------------

    st.subheader(
        "Field detection by checkpoint"
    )

    detail_rows = []

    for stage in STAGES:
        if stage not in outputs:
            continue

        response = str(
            outputs[stage].get("response") or ""
        )

        for field in LIVE_REPORT_FIELDS:
            detected = live_field_detected(response, field)

            detail_rows.append(
                {
                    "Model": stage,
                    "Report field": field,
                    "Detected": (
                        "Yes" if detected else "No"
                    ),
                }
            )

    detail_df = pd.DataFrame(detail_rows)

    with st.expander(
        "View individual report-field checks"
    ):
        st.dataframe(
            detail_df,
            hide_index=True,
            use_container_width=True,
        )

    # -------------------------
    # Downloads
    # -------------------------

    st.download_button(
        "Download live metrics (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="live_qwen_metrics.csv",
        mime="text/csv",
        key="download_live_metrics",
    )

    st.info(
        "These charts describe the current generation only. "
        "They do not assign evidence recall, hallucination rates "
        "or expert-reviewed scores to newly entered notes."
    )


# ============================================================
# 5. OFFLINE CHECKPOINT SUMMARY (NO HOSTED MODEL REQUESTS)
# ============================================================

def local_checkpoint_explanation(
    case_id: str,
    source: str,
    subset: pd.DataFrame,
    evidence: pd.DataFrame,
) -> str:
    """Summarize the saved review labels without consuming external AI credits."""
    source_items = evidence[evidence["id"] == case_id]
    lines = [
        "### Evidence review (offline, rule-based)",
        "This summary uses the saved review labels—not another model or new inference.",
    ]
    for stage in STAGES:
        if stage not in subset.index:
            continue
        judged = source_items[source_items["stage"] == stage]
        preserved = judged.loc[judged["verified_present"] == 1, "evidence"].tolist()
        missed = judged.loc[judged["verified_present"] == 0, "evidence"].tolist()
        r = subset.loc[stage]
        flags = [
            label for col, label in [
                ("unsupported_observation", "unsupported observation"),
                ("unsupported_interpretation", "unsupported interpretation"),
                ("truncated", "truncated response"),
            ] if int(r[col]) == 1
        ]
        lines.append(
            f"\n**{stage}:** {len(preserved)}/{len(judged)} evidence items preserved. "
            f"Flags: {', '.join(flags) if flags else 'none recorded'}."
        )
        if missed:
            lines.append(
                "- Missing or distorted: " + "; ".join(map(str, missed[:5]))
                + (f"; and {len(missed) - 5} more" if len(missed) > 5 else "")
            )
    lines.append(
        "\n**Caveat:** The review labels are AI-assisted; "
        "these descriptive judgments are not independently verified industrial assessments."
    )
    return "\n".join(lines)


# ============================================================
# 6. INITIALIZE DASHBOARD
# ============================================================

try:
    evidence, errors = load_data()
    reports, stage_scores = compute_scores(
        evidence,
        errors,
    )

except (
    ValueError,
    OSError,
    pd.errors.ParserError,
) as ex:
    st.error(
        f"Cannot load evaluation data: {ex}"
    )
    st.stop()


scenario_ids = list(
    dict.fromkeys(errors["id"])
)


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🩵 Maintenance AI Lab")

    st.caption(
        "Qwen2.5-1.5B · 4 checkpoints · "
        "saved benchmark + live inference"
    )

    page = st.radio(
        "Explore",
        [
            "Research overview",
            "Checkpoint comparison",
            "Evidence inspector",
            "Try a new note",
            "Methods & limits",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.info(
        "The benchmark reports are **saved outputs from "
        "real model runs**. The Try a new note tab can "
        "run your fine-tuned Qwen checkpoints when a "
        "secured GPU backend is connected. "
        "New results do not inherit benchmark scores."
    )

    st.caption(
        "Data: 10 synthetic cases per model; "
        "AI-assisted review labels, not independent "
        "expert certification."
    )


# ============================================================
# 8. RESEARCH OVERVIEW
# ============================================================

if page == "Research overview":

    st.markdown(
        """
        <section class="hero-panel">
          <div class="hero-eyebrow">INDUSTRIAL AI · DOCUMENTATION ASSISTANT</div>
          <div class="hero-title">Industrial Maintenance Document Assistant</div>
          <div class="hero-subtitle">An Experimental Analysis of LLM Post-Training and Fine-Tuning</div>
          <p>Turn raw equipment observations into clearer draft maintenance reports, while comparing domain-adapted AI models.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.header("For maintenance, reliability and operations teams")
    st.write(
        "Technician observations are often recorded as unstructured notes. "
        "This experimental assistant helps convert them into structured **draft maintenance reports** "
        "that are easier to review and compare. It helps document measurements, symptoms, "
        "actions already taken and unknowns, but **does not diagnose equipment faults**."
    )

    st.subheader("How to use the application")
    usage_steps = [
        ("01 · Enter a technician note",
         "Paste a fictional or non-confidential observation with equipment ID, measurements, symptoms, timestamps, actions and unknowns."),
        ("02 · Choose checkpoints",
         "Try SFT for a structured draft, or select Base Qwen, CPT, SFT and DPO to compare different post-training stages."),
        ("03 · Generate reports",
         "Open 'Try a new note' and click Generate. Live results require the RunPod GPU backend to be online."),
        ("04 · Review and download",
         "Verify facts against the original note, compare the reports and charts, then download JSON outputs or CSV metrics."),
    ]
    for i in range(0, len(usage_steps), 2):
        cols = st.columns(2)
        for col, (heading, description) in zip(cols, usage_steps[i:i+2]):
            with col:
                st.markdown(
                    f'<div class="soft-card"><strong>{heading}</strong><p>{description}</p></div>',
                    unsafe_allow_html=True,
                )

    st.info(
        "**Example fictional input:** Gas compressor K-204 experienced increased vibration. "
        "Discharge temperature reached 96°C, suction pressure remained steady at 18 bar, "
        "and the unit was placed on standby at 14:35. The root cause remains unconfirmed."
    )
    st.warning(
        "Educational documentation prototype only. Do not use real confidential plant records, "
        "authorize maintenance, or treat AI output as a certified QA/QC report."
    )

    st.subheader("Explore the application")
    st.markdown(
        """
        | Section | What it does |
        |---|---|
        | **Checkpoint comparison** | Side-by-side saved model outputs for the same scenario, with review flags and evidence recall. |
        | **Evidence & error inspector** | See which source observations were preserved or omitted, plus AI-assisted error flags. |
        | **Try your own maintenance note** | Generate fresh draft reports with selected Qwen checkpoints; compare latency, token use and report structure. |
        | **Research overview** | Learn the use case, training stages, and saved evaluation results. |
        | **Methods & limits** | Read the scoring definitions, caveats, and reproducibility details. |
        """
    )
    st.caption(
        "Saved benchmark pages work while the GPU is offline. New generations require a running RunPod backend."
    )

    st.divider()
    st.header("Behind the AI: an experiment in post-training")
    st.write(
        "This project compares four Qwen2.5-1.5B checkpoints to investigate whether additional "
        "training improves format, preservation of source evidence, and restraint around uncertain claims."
    )
    st.markdown(
        '<div class="stage-flow"><strong>Base Qwen</strong> → '
        '<strong>CPT</strong> (domain text) → '
        '<strong>SFT</strong> (example reports) → '
        '<strong>DPO</strong> (chosen/rejected preferences)<br>'
        '<small>Each saved checkpoint is evaluated on the same test scenarios.</small></div>',
        unsafe_allow_html=True,
    )
    with st.expander("What do Base Qwen, CPT, SFT and DPO mean?", expanded=True):
        st.markdown(
            """
            **Base Qwen — Starting model:** General language model before this project's maintenance-specific training.

            **CPT (Continued Pretraining) — Domain adaptation:** Additional next-token training on industrial maintenance text helps the model learn domain terminology and patterns.

            **SFT (Supervised Fine-Tuning) — Structured instructions:** Trains on example technician-note → report pairs to encourage consistent reporting.

            **DPO (Direct Preference Optimization) — Preferred responses:** Trains using chosen/rejected response pairs, encouraging cautious and evidence-aware outputs. Unlike PPO-based RLHF, this does not require a separate reward model.
            """
        )
        st.caption("These are training objectives, not guarantees of better factual accuracy.")
    with st.expander("LoRA and QLoRA — how were the models trained efficiently?"):
        st.write(
            "LoRA learns small, low-rank adapter matrices instead of updating all base-model weights. "
            "QLoRA uses quantized base weights alongside trainable adapters to reduce memory requirements."
        )
        st.code("Frozen base model (W₀) + low-rank update (B × A) → adapted checkpoint", language=None)
        st.caption(
            "SFT learns from a target answer; DPO learns from which of two answers is preferred."
        )

    st.divider()
    st.header("Experimental evaluation — saved benchmark")
    st.caption(
        "10 synthetic scenarios per checkpoint (40 reports); AI-assisted review labels. "
        "Historical evaluation is separate from newly generated live responses."
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Saved reports",
        f"{len(reports)}",
    )

    b.metric(
        "Test scenarios",
        f"{reports['id'].nunique()}",
    )

    c.metric(
        "Evidence judgments",
        f"{len(evidence)}",
    )

    d.metric(
        "Checkpoints",
        f"{reports['stage'].nunique()}",
    )

    st.subheader(
        "Evidence preservation vs report structure"
    )

    chart = (
        stage_scores.set_index("stage")[
            ["recall", "field_coverage"]
        ]
        * 100
    )

    st.bar_chart(
        chart.rename(
            columns={
                "recall": "Evidence recall (%)",
                "field_coverage": (
                    "Required-field presence (%)"
                ),
            }
        ),
        stack=False,
    )

    st.caption(
        "Evidence recall = mean of per-scenario reviewed "
        "recall (each scenario weighted equally). "
        "Required-field presence measures seven heading "
        "labels, not correctness of their content."
    )

    left, right = st.columns(
        [1.45, 1]
    )

    with left:
        st.subheader("Results table")

        table = stage_scores.copy()

        table["Evidence recall"] = (
            (table["recall"] * 100)
            .round(1)
            .astype(str)
            + "%"
        )

        table["Field presence"] = (
            (table["field_coverage"] * 100)
            .round(1)
            .astype(str)
            + "%"
        )

        st.dataframe(
            table[
                [
                    "stage",
                    "Evidence recall",
                    "Field presence",
                    "full_fields",
                    "unsupported_observations",
                    "unsupported_interpretations",
                    "truncated",
                ]
            ].rename(
                columns={
                    "stage": "Checkpoint",
                    "full_fields": (
                        "All 7 fields (of 10)"
                    ),
                    "unsupported_observations": (
                        "Invented observations"
                    ),
                    "unsupported_interpretations": (
                        "Unsupported interpretations"
                    ),
                    "truncated": "Truncated reports",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.subheader(
            "Failure-mode counts"
        )

        err_chart = (
            stage_scores.set_index("stage")[
                [
                    "unsupported_observations",
                    "unsupported_interpretations",
                    "truncated",
                ]
            ]
        )

        st.bar_chart(
            err_chart.rename(
                columns={
                    "unsupported_observations": (
                        "Invented observation"
                    ),
                    "unsupported_interpretations": (
                        "Unsupported interpretation"
                    ),
                    "truncated": "Truncated",
                }
            ),
            stack=False,
        )

    st.subheader(
        "What the experiment suggests"
    )

    st.markdown(
        """
- **Base Qwen** preserved the largest share of source evidence but often wrote long, speculative reports; three outputs ended prematurely.
- **SFT** achieved seven-field presence in all ten test reports, but retained fewer source observations.
- **DPO** retained more observations than SFT in this benchmark, yet produced some unsupported observations and interpretations.
- **CPT** did not demonstrate a uniform gain in downstream reliability; this pipeline alone does not isolate CPT's causal contribution.
"""
    )

    st.warning(
        "These are small-sample descriptive findings "
        "from synthetic cases with AI-assisted review "
        "labels. They do not establish industrial safety "
        "or statistically significant model superiority."
    )

    with st.expander("Additional insights from the saved test set and live demos"):
        st.markdown(
            """
            - **Format versus evidence:** SFT and DPO achieved full seven-heading presence in the saved benchmark,
              while Base Qwen preserved more source observations.
            - **DPO trade-off:** Evidence preservation was 58.3% for DPO versus 42.7% for SFT,
              but 3/10 DPO reports were flagged for unsupported observations.
            - **CPT limitation:** Without a direct Base → SFT control, the study cannot isolate CPT's contribution.
            - **Live compressor K-204:** Base Qwen explicitly retained the 2.8 → 6.4 mm/s vibration
              rise and 82 → 96°C temperature change; a CPT response introduced an unsupported statement
              about load/cooling conditions.
            - **Live pump P-642:** An SFT response followed the report format but omitted some original
              operating details and uncertainty qualifiers.
            """
        )
        st.caption(
            "Live demos are qualitative examples, not additional expert-reviewed benchmark scores."
        )


# ============================================================
# 9. CHECKPOINT COMPARISON
# ============================================================

elif page == "Checkpoint comparison":

    st.title("Checkpoint comparison")

    selected = st.selectbox(
        "Choose benchmark scenario",
        scenario_ids,
    )

    subset = reports[
        reports["id"] == selected
    ].set_index("stage")

    source = subset.iloc[0][
        "technician_note"
    ]

    st.subheader(
        "Technician note (source)"
    )

    st.info(source)

    st.caption(
        "Same original technician note supplied to "
        "each checkpoint with a fixed inference template "
        "and greedy decoding."
    )

    selected_stages = st.multiselect(
        "Checkpoints to compare",
        STAGES,
        default=STAGES,
    )

    if selected_stages:
        for offset in range(
            0,
            len(selected_stages),
            2,
        ):
            cols = st.columns(2)

            for col, checkpoint in zip(
                cols,
                selected_stages[offset:offset + 2],
            ):
                with col:
                    if checkpoint not in subset.index:
                        st.warning(
                            f"No saved {checkpoint} output"
                        )
                        continue

                    row = subset.loc[checkpoint]

                    st.markdown(
                        f"#### {checkpoint}"
                    )

                    st.caption(
                        f"Evidence recall "
                        f"{row['evidence_recall']:.0%} · "
                        f"{int(row['required_fields_present'])}"
                        "/7 fields detected"
                    )

                    show_report(
                        row["model_response"]
                    )

                    if row["truncated"]:
                        st.warning(
                            "Report appears truncated"
                        )

                    if row["unsupported_observation"]:
                        st.error(
                            "Flagged unsupported observation"
                        )

                    if row["unsupported_interpretation"]:
                        st.warning(
                            "Flagged unsupported interpretation"
                        )

    st.divider()

    st.subheader("Evidence summary (offline, no inference credits)")

    st.caption(
        "Optional rule-based summary of the saved evidence-review labels. "
        "This does not call Hugging Face, Qwen, or any external inference service."
    )

    with st.expander("Show saved evidence summary"):
        st.markdown(
            local_checkpoint_explanation(selected, source, subset, evidence)
        )

    st.download_button(
        "Download this scenario's model comparisons (CSV)",
        reports[
            reports["id"] == selected
        ].to_csv(index=False).encode("utf-8"),
        file_name=f"comparison_{selected}.csv",
        mime="text/csv",
    )


# ============================================================
# 10. EVIDENCE INSPECTOR
# ============================================================

elif page == "Evidence inspector":

    st.title(
        "Evidence & error inspector"
    )

    selected = st.selectbox(
        "Benchmark scenario",
        scenario_ids,
    )

    st.caption(
        "✓ = preserved with important qualifiers; "
        "× = omitted, contradicted or materially distorted. "
        "These are AI-assisted review judgments."
    )

    case_evidence = evidence[
        evidence["id"] == selected
    ]

    evidence_pivot = (
        case_evidence.pivot_table(
            index="evidence",
            columns="stage",
            values="verified_present",
            aggfunc="first",
        )
        .reindex(columns=STAGES)
    )

    st.dataframe(
        evidence_pivot.replace(
            {
                1: "✓ Preserved",
                0: "× Not preserved",
            }
        ),
        use_container_width=True,
    )

    selected_stage = st.selectbox(
        "Inspect review notes for checkpoint",
        STAGES,
    )

    selected_evidence = case_evidence[
        case_evidence["stage"] == selected_stage
    ]

    for _, record in selected_evidence.iterrows():
        status = (
            "✓"
            if record["verified_present"]
            else "×"
        )

        with st.expander(
            f"{status} {record['evidence']}"
        ):
            note = record.get(
                "reviewer_notes"
            )

            st.write(
                note
                if pd.notna(note)
                and str(note).strip()
                else "No additional reviewer note recorded."
            )

    case_error = reports[
        (reports["id"] == selected)
        & (reports["stage"] == selected_stage)
    ].iloc[0]

    st.subheader(
        "Report-level findings"
    )

    x, y, z = st.columns(3)

    x.metric(
        "Invented observation",
        (
            "Yes"
            if case_error["unsupported_observation"]
            else "No"
        ),
    )

    y.metric(
        "Unsupported interpretation",
        (
            "Yes"
            if case_error["unsupported_interpretation"]
            else "No"
        ),
    )

    z.metric(
        "Truncation",
        (
            "Yes"
            if case_error["truncated"]
            else "No"
        ),
    )

    note = case_error.get(
        "reviewer_notes"
    )

    st.info(
        str(note)
        if pd.notna(note)
        and str(note).strip()
        else "No further error-review notes recorded."
    )

    st.subheader(
        "How this assessment is formed"
    )

    st.write(
        f"The {selected_stage} response retained "
        f"**{int(selected_evidence['verified_present'].sum())} "
        f"of {len(selected_evidence)}** checklist items "
        f"for {selected}. "
        "The error flags are separate report-level "
        "judgments: an omitted observation is not "
        "the same as an invented observation, and a "
        "clearly qualified hypothesis is not "
        "automatically an unsupported claim."
    )


# ============================================================
# 11. LIVE INFERENCE + NEW COMPARISON CHARTS
# ============================================================

elif page == "Try a new note":

    st.title(
        "Try your own maintenance note"
    )

    st.caption(
        "Live inference loads your Qwen2.5-1.5B "
        "Base/CPT/SFT/DPO checkpoints on a separate "
        "GPU backend. This does not use Phi-4 or "
        "reassign benchmark metrics to new notes."
    )

    typed = st.text_area(
        "Technician note",
        height=190,
        key="live_note",
        placeholder=(
            "e.g., Pump P-218 shuddered "
            "twice after startup..."
        ),
        max_chars=3500,
    )

    chosen = st.multiselect(
        "Checkpoints to run",
        STAGES,
        default=[
            "CPT",
            "SFT",
            "DPO",
        ],
        help=(
            "Each checkpoint is loaded sequentially "
            "on the GPU, which may take time."
        ),
    )

    max_tokens = st.slider(
        "Maximum output tokens per checkpoint",
        128,
        512,
        350,
        step=32,
    )

    # -------------------------
    # Streamlit Secrets
    # -------------------------

    # Separate Serverless demo settings. Original normal-Pod app unchanged.
    endpoint_id = os.getenv("SERVERLESS_ENDPOINT_ID", "")
    api_key = os.getenv("SERVERLESS_API_KEY", "")
    try:
        endpoint_id = st.secrets.get("SERVERLESS_ENDPOINT_ID", endpoint_id)
        api_key = st.secrets.get("SERVERLESS_API_KEY", api_key)
    except (FileNotFoundError, KeyError):
        pass
    if not endpoint_id or not api_key:
        st.info(
            "Configure SERVERLESS_ENDPOINT_ID and SERVERLESS_API_KEY in Streamlit Secrets. "
            "Saved benchmark comparisons remain available."
        )

    # -------------------------
    # Request identifier
    # -------------------------

    request_id = hashlib.sha256(
        json.dumps(
            {
                "note": typed.strip(),
                "stages": chosen,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    # -------------------------
    # RunPod inference
    # -------------------------

    if st.button(
        "Generate with my fine-tuned Qwen models",
        type="primary",
        disabled=(
            not bool(endpoint_id and api_key)
            or len(typed.strip()) < 20
            or not chosen
        ),
    ):

        with st.spinner(
            "Serverless is allocating GPU compute "
            "and generating your reports…"
        ):

            try:
                progress = st.empty()
                result = run_serverless_job(
                    endpoint_id=str(endpoint_id).strip(),
                    api_key=str(api_key).strip(),
                    note=typed.strip(),
                    stages=chosen,
                    max_new_tokens=max_tokens,
                    on_status=lambda message: progress.info(message),
                )
                progress.empty()
                if not isinstance(result.get("outputs"), dict):
                    raise ValueError("Serverless returned an invalid result.")
                st.session_state["live_result"] = {
                    "request_id": request_id,
                    "result": result,
                }

            except (
                requests.RequestException,
                RuntimeError,
                ValueError,
            ) as ex:
                st.error(
                    f"Live inference unavailable: {ex}"
                )

    # -------------------------
    # Display actual results
    # -------------------------

    cached = st.session_state.get(
        "live_result"
    )

    if (
        cached
        and cached["request_id"] == request_id
    ):

        result = cached["result"]

        st.success(
            "Generated using your actual "
            "Qwen checkpoint weights"
        )

        st.warning(
            "Exploratory, unreviewed generations. "
            "Benchmark recall and failure-rate scores "
            "do NOT apply to this newly entered note. "
            "Do not use for maintenance decisions."
        )

        outputs = result["outputs"]

        display_stages = [stage for stage in STAGES if stage in chosen]

        for i in range(
            0,
            len(display_stages),
            2,
        ):
            columns = st.columns(2)

            for column, stage in zip(
                columns,
                display_stages[i:i + 2],
            ):

                if stage not in outputs:
                    continue

                item = outputs[stage]

                with column:
                    st.markdown(
                        f'<div style="background:{MODEL_COLORS[stage]};border-radius:9px;'
                        'padding:7px 12px;color:#173E63;font-weight:700;margin-bottom:9px">'
                        f'{stage}</div>',
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        f"{item.get('generated_tokens', '?')} "
                        "generated tokens · "
                        f"{item.get('latency_seconds', '?')} s"
                    )

                    show_report(
                        item.get(
                            "response",
                            "",
                        )
                    )

                    if item.get(
                        "reached_token_limit"
                    ):
                        st.warning(
                            "Reached generation token limit; "
                            "the response may be incomplete."
                        )

        # ================================================
        # NEW: LIVE MODEL COMPARISON VISUALIZATIONS
        # ================================================

        show_live_comparison(outputs)

        # -------------------------
        # Download original outputs
        # -------------------------

        st.download_button(
            "Download live checkpoint outputs (JSON)",
            data=json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            file_name="live_qwen_comparison.json",
            mime="application/json",
        )

    elif (
        cached
        and cached["request_id"] != request_id
    ):
        st.caption(
            "Input or checkpoint selection changed. "
            "Generate again to view results for this request."
        )

    st.divider()

    st.caption(
        "Serverless Flex workers scale to zero when idle. Cold starts and GPU "
        "capacity may delay inference. Saved benchmark views work without a GPU."
    )


# ============================================================
# 12. METHODS AND LIMITATIONS
# ============================================================

else:

    st.title(
        "Methods, caveats & reproducibility"
    )

    st.subheader(
        "The experiment"
    )

    st.write(
        "Qwen2.5-1.5B was evaluated as a base checkpoint "
        "and after continued pretraining (CPT), supervised "
        "instruction fine-tuning (SFT), and direct preference "
        "optimization (DPO). The reports in this app are "
        "saved generated outputs, not reimplementations "
        "or invented examples."
    )

    st.subheader(
        "What each metric means"
    )

    st.markdown(
        """
- **Evidence recall:** a strict 0/1 judgment for each required source observation, averaged within each scenario and then equally across ten scenarios. A paraphrase can count, but key qualifiers and measurements must be retained.
- **Required-field presence:** proportion of seven section labels appearing in the output. All-seven presence is a separate binary indicator. Neither guarantees correct content or order.
- **Unsupported observation:** a new factual assertion presented as observed source evidence but not supplied by the note.
- **Unsupported interpretation:** an unwarranted claim or unwarranted certainty about a technical condition. Qualified hypotheses are treated separately.
- **Truncation:** the visible report ends prematurely; the cause cannot be established just from the output text.
"""
    )

    st.subheader(
        "Live comparison metrics"
    )

    st.markdown(
        """
- **Latency:** the time reported by the GPU backend for each checkpoint. This may include loading and inference overhead.
- **Generated tokens:** the number of output tokens reported by the backend.
- **Tokens per second:** generated tokens divided by reported latency. This is a throughput proxy if latency includes model loading.
- **Live report fields:** the number of headings detected from the live maintenance-report format, out of seven.
- **Live field coverage:** detected live headings divided by seven. It is a structural metric, not a quality score.
- **Historical benchmark scores and new live inference metrics are separate.** Live responses have not undergone the saved benchmark's evidence review.
"""
    )

    st.subheader("Limitations")

    st.markdown(
        """
1. Only **10 synthetic scenarios** are tested for each checkpoint.
2. The annotations are **AI-assisted, single-reviewer judgments** pending independent confirmation.
3. Equipment IDs were screened for direct reuse, but semantic overlap with synthetic training templates may remain.
4. Results are descriptive and should not be interpreted as statistical proof of superiority.
5. A direct SFT-from-base control would be required to isolate the effect of CPT.
6. The tool does not authorize maintenance decisions, failure diagnosis, or equipment operation.
7. Live performance charts describe individual inference runs and may vary with GPU conditions, loading order and output length.
"""
    )

    st.subheader(
        "Download reproducible review data"
    )

    st.download_button(
        "Evidence judgments (CSV)",
        evidence.to_csv(
            index=False
        ).encode(),
        "evidence_reviewed.csv",
        mime="text/csv",
    )

    st.download_button(
        "Error judgments (CSV)",
        errors.to_csv(
            index=False
        ).encode(),
        "error_reviewed.csv",
        mime="text/csv",
    )

    st.download_button(
        "Computed case-level scores (CSV)",
        reports.to_csv(
            index=False
        ).encode(),
        "case_scores.csv",
        mime="text/csv",
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "LLM Post-Training Lab · Independent educational "
    "experiment · Saved benchmark and optional Serverless "
    "Qwen adapter inference · "
    "Do not use for maintenance or safety decisions"
)
