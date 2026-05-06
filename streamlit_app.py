"""
TrialGraph — Clinical Trials Intelligence Platform
=========================================================
Single-page application with two agent panels:
  Left  — Chat Agent  (conversational Q&A via Ollama + GraphRAG)
  Right — Analysis Agent (auto-generates plots + summaries for every query)

Run:  streamlit run streamlit_app.py
Deps: pip install streamlit neo4j networkx pandas matplotlib seaborn requests python-louvain
"""

import re
import json
import textwrap
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from application import (
    KGClient,
    drug_evidence, drug_competition, drug_geo, drug_paths,
    disease_landscape, disease_design, disease_phase_progression,
    disease_enrollment, disease_sponsor_diversity,
    sponsor_portfolio, sponsor_geo, sponsor_pipeline,
    sponsor_drugs, sponsor_collaborators,
    centrality, community_detection, drug_repurposing,
    trial_density, trial_timeline, geo_phase_heatmap,
    build_graphrag_context,
)
from network_graph_supplementary import (
    drug_condition_subgraph,
    sponsor_network,
    condition_similarity_network,
    graph_metrics,
    degree_distribution,
    bridge_drugs,
)

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TrialGraph",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ════════════════════════════════════════
   ROOT TOKENS
════════════════════════════════════════ */
:root {
    --bg-base:       #070a12;
    --bg-surface:    #0b0f1c;
    --bg-raised:     #0f1424;
    --bg-hover:      #131829;
    --bg-active:     #171e30;
    --border-subtle: rgba(255,255,255,0.055);
    --border-mid:    rgba(255,255,255,0.09);
    --border-strong: rgba(255,255,255,0.14);
    --accent:        #4f7cff;
    --accent-dim:    rgba(79,124,255,0.15);
    --accent-glow:   rgba(79,124,255,0.08);
    --text-primary:  #e4eaf8;
    --text-secondary:#a0aec8;
    --text-muted:    #6b7fa8;
    --text-faint:    #4a5878;
    --green:  #22d3a5;  --green-bg:  rgba(34,211,165,0.08);
    --red:    #f87171;  --red-bg:    rgba(248,113,113,0.08);
    --amber:  #fbbf24;  --amber-bg:  rgba(251,191,36,0.08);
    --font-ui:   'Inter', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --r-sm: 6px;  --r-md: 10px;  --r-lg: 14px;
}

/* ════════════════════════════════════════
   GLOBAL RESETS — force Inter everywhere
════════════════════════════════════════ */
html, body, [class*="css"], .stApp, .stMarkdown,
.stTextInput input, .stSelectbox, button, label, p, span, div {
    font-family: var(--font-ui) !important;
}

/* ════════════════════════════════════════
   APP CHROME
════════════════════════════════════════ */
.stApp,
[data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}
/* Hide the Streamlit top toolbar decoration */
[data-testid="stDecoration"] { display: none !important; }

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] .block-container { padding: 0 !important; }
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* Sidebar labels */
[data-testid="stSidebar"] label > div > p,
[data-testid="stSidebar"] label {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="password"] {
    background: var(--bg-base) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 0.7rem !important;
    transition: border-color 0.15s !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--bg-base) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
}

/* ════════════════════════════════════════
   MAIN CONTENT AREA
════════════════════════════════════════ */
.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 100% !important;
}

/* ════════════════════════════════════════
   TYPOGRAPHY
════════════════════════════════════════ */
h1, h2, h3, h4 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}
p, li, .stMarkdown p { color: var(--text-secondary); }
code, .stCode {
    font-family: var(--font-mono) !important;
    background: var(--bg-raised) !important;
    color: #7eb8f7 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-size: 0.8em !important;
    border: 1px solid var(--border-subtle) !important;
}

/* ════════════════════════════════════════
   BUTTONS
════════════════════════════════════════ */
.stButton > button {
    background: var(--bg-raised) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-sm) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.15s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--border-mid) !important;
    color: var(--text-primary) !important;
}
.stButton > button:active {
    background: var(--bg-active) !important;
    transform: translateY(1px);
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: transparent !important;
    color: #fff !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #6b8fff !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}

/* ════════════════════════════════════════
   MAIN INPUTS
════════════════════════════════════════ */
.stTextInput input, .stTextArea textarea {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
.stSelectbox > div > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text-primary) !important;
}

/* ════════════════════════════════════════
   CHAT INPUT
════════════════════════════════════════ */
[data-testid="stChatInput"] > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--r-md) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.88rem !important;
}
[data-testid="stChatInput"]:focus-within > div {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ════════════════════════════════════════
   CHAT MESSAGES
════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    border-radius: var(--r-md) !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem !important;
}
/* Assistant — deep navy with blue left stripe */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #07101f !important;
    border: 1px solid rgba(79,124,255,0.14) !important;
    border-left: 3px solid var(--accent) !important;
}
/* User — warm dark with amber left stripe */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #110d06 !important;
    border: 1px solid rgba(201,123,42,0.14) !important;
    border-left: 3px solid #c97b2a !important;
}
[data-testid="chatAvatarIcon-assistant"] svg { color: var(--accent) !important; }
[data-testid="chatAvatarIcon-user"]      svg { color: #c97b2a !important; }

/* ════════════════════════════════════════
   EXPANDERS
════════════════════════════════════════ */
details {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-sm) !important;
    overflow: hidden !important;
}
details > summary {
    font-family: var(--font-ui) !important;
    font-size: 0.74rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 0.75rem !important;
    cursor: pointer !important;
    list-style: none !important;
    transition: color 0.12s !important;
}
details > summary:hover { color: var(--text-secondary) !important; }
details[open] > summary { color: var(--accent) !important; border-bottom: 1px solid var(--border-subtle); }

/* ════════════════════════════════════════
   DATAFRAMES
════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-md) !important;
    overflow: hidden !important;
}

/* ════════════════════════════════════════
   CODE BLOCKS
════════════════════════════════════════ */
.stCodeBlock {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-sm) !important;
    overflow: hidden !important;
}
.stCodeBlock pre {
    background: var(--bg-base) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

/* ════════════════════════════════════════
   CHECKBOX
════════════════════════════════════════ */
.stCheckbox label p {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}
.stCheckbox:hover label p { color: var(--text-primary) !important; }

/* ════════════════════════════════════════
   SPINNER
════════════════════════════════════════ */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ════════════════════════════════════════
   ALERTS
════════════════════════════════════════ */
.stAlert {
    border-radius: var(--r-sm) !important;
    border-left-width: 3px !important;
    background: var(--bg-raised) !important;
}

/* ════════════════════════════════════════
   HR
════════════════════════════════════════ */
hr { border: none !important; border-top: 1px solid var(--border-subtle) !important; margin: 0.8rem 0 !important; }

/* ════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
════════════════════════════════════════ */

/* Status pills */
.status-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em;
    padding: 3px 10px; border-radius: 999px;
    font-family: var(--font-ui);
}
.status-ok   { background: var(--green-bg); color: var(--green); border: 1px solid rgba(34,211,165,0.2); }
.status-err  { background: var(--red-bg);   color: var(--red);   border: 1px solid rgba(248,113,113,0.2); }
.status-warn { background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(251,191,36,0.2); }
.dot { width: 5px; height: 5px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-ok { background: var(--green); box-shadow: 0 0 6px var(--green); }
.dot-err  { background: var(--red); }
.dot-warn { background: var(--amber); }

/* Sidebar section label */
.sidebar-section {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--text-faint);
    padding: 0.8rem 0 0.35rem; margin-top: 0.1rem;
    border-top: 1px solid var(--border-subtle);
    font-family: var(--font-ui);
}

/* Panel headers */
.panel-header {
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.64rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; padding-bottom: 0.7rem;
    margin-bottom: 1rem; font-family: var(--font-ui);
    border-bottom: 1px solid var(--border-subtle);
}
.panel-header-chat     { color: #3d5c9e; }
.panel-header-analysis { color: #2a7060; }
.panel-tag {
    font-size: 0.58rem; font-weight: 500; letter-spacing: 0.08em;
    text-transform: none; color: var(--text-faint);
}

/* Turn labels in analysis panel */
.turn-label-latest {
    border-left: 2px solid var(--accent);
    padding: 4px 0 4px 10px; margin: 1.2rem 0 0.6rem;
}
.turn-label-prior {
    border-left: 2px solid var(--border-subtle);
    padding: 4px 0 4px 10px; margin: 1.2rem 0 0.6rem;
}
.turn-num {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--text-faint);
    font-family: var(--font-ui);
}
.turn-q {
    font-size: 0.8rem; color: var(--text-secondary);
    font-style: italic; margin-top: 2px; font-family: var(--font-ui);
}

/* Analysis panel chart title */
.chart-title {
    font-size: 0.77rem; font-weight: 600; color: var(--text-muted);
    letter-spacing: 0.02em; margin: 0.9rem 0 0.3rem;
    font-family: var(--font-ui);
    text-transform: uppercase; font-size: 0.64rem; letter-spacing: 0.12em;
}

/* Analysis summary aside */
.analysis-summary {
    font-size: 0.8rem; color: var(--text-muted); line-height: 1.7;
    border-left: 2px solid var(--border-mid); padding-left: 0.85rem;
    margin: 0.4rem 0 0.8rem; font-family: var(--font-ui);
}

/* Empty state cards */
.empty-card {
    margin: 2.5rem 0; padding: 2rem;
    background: var(--bg-surface); border: 1px dashed var(--border-subtle);
    border-radius: var(--r-lg); text-align: center;
}
.empty-card-title {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--text-faint); margin-bottom: 0.5rem;
    font-family: var(--font-ui);
}
.empty-card-body {
    font-size: 0.82rem; color: var(--text-faint); line-height: 1.65;
    font-family: var(--font-ui);
}

/* Separator between analysis turns */
.turn-sep {
    border: none; border-top: 1px solid var(--border-subtle);
    margin: 1.5rem 0 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────

defaults = {
    "kg": None,
    "connected": False,
    "ollama_models": [],
    "chat_history": [],        # list of {role, content, meta}
    "analysis_panels": [],     # list of {turn_id, label, panels:[{title,fig,df,cypher,params}]}
    "last_entity": None,       # {entity_type, entity_value, search_token} — carried across turns
    "conv_context": {"entity_type": None, "entity_value": "", "search_token": ""},  # active conversation entity
}
# Ensure conv_context exists (may not be in defaults dict in older sessions)
if "conv_context" not in st.session_state:
    st.session_state["conv_context"] = {
        "entity_type": None, "entity_value": "", "search_token": ""
    }

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
#  OLLAMA UTILITIES
# ─────────────────────────────────────────────────────────────

def ollama_is_running(url: str) -> bool:
    try:
        return requests.get(f"{url}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False

def get_ollama_models(url: str) -> list:
    try:
        r = requests.get(f"{url}/api/tags", timeout=4)
        return [m["name"] for m in r.json().get("models", [])] if r.ok else []
    except Exception:
        return []

def ollama_stream(url: str, model: str, system: str, messages: list):
    """Yield text tokens from Ollama /api/chat (streaming)."""
    payload = {
        "model": model,
        "stream": True,
        "options": {"num_predict": -1},   # no token limit
        "messages": [{"role": "system", "content": system}] + messages,
    }
    with requests.post(f"{url}/api/chat", json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break

def ollama_complete(url: str, model: str, system: str, prompt: str) -> str:
    """Non-streaming single completion (used for summary agent)."""
    payload = {
        "model": model,
        "stream": False,
        "options": {"num_predict": -1},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    }
    r = requests.post(f"{url}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ─────────────────────────────────────────────────────────────
#  INTENT ROUTER
#  Classifies the user question and dispatches the right
#  KG queries + plots. Returns a structured result dict.
# ─────────────────────────────────────────────────────────────

# Cypher queries keyed by analysis type — stored for transparency panel

def query_nct_detail(kg: KGClient, nct_id: str) -> dict | None:
    """
    Fetch all available details for a single trial by NCT ID.
    Returns a flat dict of trial metadata, or None if not found.
    """
    rows = kg.query(CYPHER_REGISTRY["nct_detail"], {"nct_id": nct_id.upper()})
    return rows[0] if rows else None


def build_nct_context(trial: dict) -> str:
    """
    Format a trial detail dict as structured text for LLM context.
    """
    if not trial:
        return ""
    lines = [f"## Trial Detail: {trial.get('nct_id', 'Unknown')}\n"]
    fields = [
        ("Phase",             "phase"),
        ("Status",            "status"),
        ("Study Type",        "study_type"),
        ("Allocation",        "allocation"),
        ("Masking",           "masking"),
        ("Enrollment",        "enrollment"),
        ("Start Date",        "start_date"),
        ("Completion Date",   "completion_date"),
    ]
    for label, key in fields:
        val = trial.get(key)
        if val:
            lines.append(f"- {label}: {val}")
    for label, key in [("Conditions", "conditions"), ("Interventions", "interventions"),
                        ("Sponsors", "sponsors"), ("Countries", "countries"),
                        ("Arm Types", "arm_types")]:
        vals = trial.get(key, [])
        if isinstance(vals, list):
            vals = [v for v in vals if v]
        if vals:
            lines.append(f"- {label}: {', '.join(vals)}")
    return "\n".join(lines)

CYPHER_REGISTRY = {
    "drug_evidence": textwrap.dedent("""
        MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)
        WHERE toLower(i.name) CONTAINS toLower($drug)
        OPTIONAL MATCH (st)-[:STUDIES]->(c:Condition)
        OPTIONAL MATCH (st)<-[:SPONSORS]-(s:Sponsor)
        OPTIONAL MATCH (st)-[:CONDUCTED_AT]->(l:Location)
        RETURN st.nct_id AS trial, st.phases AS phase,
               st.overall_status AS status, st.enrollment AS enrollment,
               c.name AS condition, s.name AS sponsor, l.country AS country
    """).strip(),

    "drug_competition": textwrap.dedent("""
        MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)-[:STUDIES]->(c:Condition)
        MATCH (c)<-[:STUDIES]-(st2:Study)-[:USES_INTERVENTION]->(i2:Intervention)
        WHERE toLower(i.name) CONTAINS toLower($drug)
          AND toLower(i2.name) <> toLower($drug)
        RETURN i2.name AS competitor, COUNT(DISTINCT st2) AS trials
        ORDER BY trials DESC LIMIT 30
    """).strip(),

    "drug_geo": textwrap.dedent("""
        MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)-[:CONDUCTED_AT]->(l:Location)
        WHERE toLower(i.name) CONTAINS toLower($drug)
        RETURN l.country AS country, COUNT(DISTINCT st.nct_id) AS trials
        ORDER BY trials DESC
    """).strip(),

    "drug_paths": textwrap.dedent("""
        MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)
        WHERE toLower(i.name) CONTAINS toLower($drug)
        MATCH (st)-[:STUDIES]->(c:Condition)
        MATCH (st)<-[:SPONSORS]-(s:Sponsor)
        RETURN i.name AS drug, st.nct_id AS trial,
               c.name AS condition, s.name AS sponsor, s.class AS sponsor_class
    """).strip(),

    "disease_landscape": textwrap.dedent("""
        MATCH (st:Study)-[:STUDIES]->(c:Condition)
        WHERE toLower(c.name) CONTAINS toLower($disease)
        MATCH (st)-[:USES_INTERVENTION]->(i:Intervention)
        RETURN i.name AS drug, COUNT(DISTINCT st.nct_id) AS trials
        ORDER BY trials DESC LIMIT 30
    """).strip(),

    "disease_phase": textwrap.dedent("""
        MATCH (st:Study)-[:STUDIES]->(c:Condition)
        WHERE toLower(c.name) CONTAINS toLower($disease)
        RETURN st.phases AS phase, COUNT(DISTINCT st.nct_id) AS trials
        ORDER BY phase
    """).strip(),

    "disease_enrollment": textwrap.dedent("""
        MATCH (st:Study)-[:STUDIES]->(c:Condition)
        WHERE toLower(c.name) CONTAINS toLower($disease)
          AND st.enrollment IS NOT NULL
        RETURN st.nct_id AS trial, st.enrollment AS enrollment, st.phases AS phase
    """).strip(),

    "disease_design": textwrap.dedent("""
        MATCH (st:Study)-[:STUDIES]->(c:Condition)
        WHERE toLower(c.name) CONTAINS toLower($disease)
        OPTIONAL MATCH (st)-[:HAS_ARM]->(a:Arm)
        RETURN st.allocation AS allocation, st.masking AS masking,
               st.study_type AS study_type, a.type AS arm_type, COUNT(*) AS count
    """).strip(),

    "disease_sponsor_diversity": textwrap.dedent("""
        MATCH (st:Study)-[:STUDIES]->(c:Condition)
        WHERE toLower(c.name) CONTAINS toLower($disease)
        MATCH (st)<-[:SPONSORS]-(s:Sponsor)
        RETURN s.name AS sponsor, s.class AS sponsor_class,
               COUNT(DISTINCT st.nct_id) AS trials
        ORDER BY trials DESC
    """).strip(),

    "sponsor_portfolio": textwrap.dedent("""
        MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
        WHERE toLower(s.name) CONTAINS toLower($sponsor)
        MATCH (st)-[:STUDIES]->(c:Condition)
        RETURN c.name AS condition, COUNT(DISTINCT st.nct_id) AS trials
        ORDER BY trials DESC LIMIT 30
    """).strip(),

    "sponsor_pipeline": textwrap.dedent("""
        MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
        WHERE toLower(s.name) CONTAINS toLower($sponsor)
        RETURN st.phases AS phase, st.overall_status AS status,
               COUNT(DISTINCT st.nct_id) AS trials
    """).strip(),

    "sponsor_geo": textwrap.dedent("""
        MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
        WHERE toLower(s.name) CONTAINS toLower($sponsor)
        MATCH (st)-[:CONDUCTED_AT]->(l:Location)
        RETURN l.country AS country, COUNT(DISTINCT st.nct_id) AS trials
        ORDER BY trials DESC
    """).strip(),

    "sponsor_collaborators": textwrap.dedent("""
        MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
        WHERE toLower(s.name) CONTAINS toLower($sponsor)
        MATCH (st)<-[:SPONSORS]-(s2:Sponsor)
        WHERE toLower(s2.name) <> toLower($sponsor)
        RETURN s2.name AS collaborator, COUNT(DISTINCT st.nct_id) AS shared_trials
        ORDER BY shared_trials DESC LIMIT 25
    """).strip(),

    "centrality": textwrap.dedent("""
        MATCH (st:Study)-[:USES_INTERVENTION]->(i:Intervention)
        MATCH (st)-[:STUDIES]->(c:Condition)
        RETURN i.name AS drug, c.name AS condition
    """).strip(),

    "repurposing": textwrap.dedent("""
        MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)-[:STUDIES]->(c:Condition)
        RETURN i.name AS drug,
               COUNT(DISTINCT c.name) AS n_conditions,
               COUNT(DISTINCT st.nct_id) AS n_trials,
               COLLECT(DISTINCT c.name)[0..8] AS sample_conditions
        ORDER BY n_conditions DESC LIMIT 40
    """).strip(),

    "geo_density": textwrap.dedent("""
        MATCH (st:Study)-[:CONDUCTED_AT]->(l:Location)
        RETURN l.country AS country, COUNT(DISTINCT st.nct_id) AS trials
        ORDER BY trials DESC
    """).strip(),

    "trial_timeline": textwrap.dedent("""
        MATCH (st:Study)
        WHERE st.start_date IS NOT NULL
        RETURN st.start_date AS start_date, st.phases AS phase,
               st.overall_status AS status, COUNT(*) AS trials
    """).strip(),

    "nct_detail": textwrap.dedent("""
        MATCH (st:Study {nct_id: $nct_id})
        OPTIONAL MATCH (st)-[:STUDIES]->(c:Condition)
        OPTIONAL MATCH (st)-[:USES_INTERVENTION]->(i:Intervention)
        OPTIONAL MATCH (st)<-[:SPONSORS]-(s:Sponsor)
        OPTIONAL MATCH (st)-[:CONDUCTED_AT]->(l:Location)
        OPTIONAL MATCH (st)-[:HAS_ARM]->(a:Arm)
        RETURN st.nct_id          AS nct_id,
               st.phases           AS phase,
               st.overall_status   AS status,
               st.study_type       AS study_type,
               st.allocation       AS allocation,
               st.masking          AS masking,
               st.enrollment       AS enrollment,
               st.start_date       AS start_date,
               st.completion_date  AS completion_date,
               COLLECT(DISTINCT c.name) AS conditions,
               COLLECT(DISTINCT i.name) AS interventions,
               COLLECT(DISTINCT s.name) AS sponsors,
               COLLECT(DISTINCT l.country) AS countries,
               COLLECT(DISTINCT a.type)   AS arm_types
    """).strip(),
}


def _kg_confirm(kg: KGClient, entity_value: str, entity_type: str) -> dict | None:
    """
    Confirm the LLM-extracted entity exists in the KG and return the
    canonical name + search token.  Uses CONTAINS so all branches match.
    Priority: exact match first, then shortest (most specific) name.
    """
    cypher_map = {
        "sponsor": "MATCH (s:Sponsor)      WHERE toLower(s.name) CONTAINS toLower($v) RETURN s.name AS name ORDER BY size(s.name) LIMIT 20",
        "drug":    "MATCH (i:Intervention) WHERE toLower(i.name) CONTAINS toLower($v) RETURN i.name AS name ORDER BY size(i.name) LIMIT 20",
        "disease": "MATCH (c:Condition)    WHERE toLower(c.name) CONTAINS toLower($v) RETURN c.name AS name ORDER BY size(c.name) LIMIT 20",
    }
    # Try the LLM-suggested type first, then fall through to others
    type_order = [entity_type] + [t for t in ("sponsor", "drug", "disease") if t != entity_type]
    vl = entity_value.lower()
    for et in type_order:
        cypher = cypher_map.get(et)
        if not cypher:
            continue
        rows = kg.query(cypher, {"v": entity_value})
        if not rows:
            continue
        names  = [r["name"] for r in rows]
        exact  = [n for n in names if n.lower() == vl]
        chosen = exact[0] if exact else min(names, key=len)
        return {
            "entity_type":  et,
            "entity_value": chosen,        # canonical KG name
            "search_token": entity_value,  # original LLM-extracted value for broad CONTAINS
        }
    return None


def llm_extract_entity(
    question: str,
    ollama_url: str,
    ollama_model: str,
    conversation_history: list | None = None,
) -> dict | None:
    """
    Use a local LLM to extract the named entity from the question.

    Returns {"entity_type": "drug|disease|sponsor|network|geo", "entity_value": "<name>"}
    or None if no entity is present.

    The LLM handles all natural language variation — verb forms, pronouns,
    phrasing differences — without any stopword lists or regex heuristics.
    """
    # Include recent conversation so the LLM understands follow-up references
    history_str = ""
    if conversation_history:
        recent = conversation_history[-4:]
        pairs  = []
        for m in recent:
            role = "User" if m["role"] == "user" else "Assistant"
            pairs.append(f"{role}: {m['content'][:300]}")
        if pairs:
            history_str = "\n\nRecent conversation:\n" + "\n".join(pairs)

    system = (
        "You are an entity extractor for a clinical trials knowledge graph. "
        "Your only job is to identify the primary named entity in the user's question "
        "and classify it. You must respond with ONLY valid JSON — no explanation, "
        "no markdown, no preamble. "
        "\n\nEntity types:"
        "\n- \"drug\": a specific drug, compound, molecule, or intervention name"
        "\n- \"disease\": a disease, condition, or medical indication"
        "\n- \"sponsor\": a pharmaceutical company, biotech, or research organisation"
        "\n- \"network\": graph-level analysis with no specific named entity"
        "\n- \"geo\": geographic or temporal analysis with no specific named entity"
        "\n\nIf a named entity is present return: "
        "{\"entity_type\": \"<type>\", \"entity_value\": \"<exact name as written>\"}"
        "\nIf no named entity is present return: null"
        "\nIf the question refers to a previously mentioned entity via pronouns "
        "(\'their\', \'its\', \'the company\', \'that drug\', etc.) return: "
        "{\"entity_type\": null, \"entity_value\": null}"
    )

    user_prompt = f"Question: {question}{history_str}"

    try:
        payload = {
            "model": ollama_model,
            "stream": False,
            "options": {"num_predict": 80, "temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
        }
        resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=20)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()

        # Strip markdown fences if the model added them
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()

        if raw.lower() in ("null", "none", ""):
            return None

        parsed = json.loads(raw)

        if not isinstance(parsed, dict):
            return None
        if parsed.get("entity_value") is None:
            return None

        et = parsed.get("entity_type", "").lower()
        ev = str(parsed.get("entity_value", "")).strip()

        if not ev or et not in ("drug", "disease", "sponsor", "network", "geo"):
            return None

        return {"entity_type": et, "entity_value": ev}

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
#  LLM-BASED ANALYSIS SELECTOR (Option 3)
#  Entity is KG-grounded. Analysis selection is LLM-driven.
# ─────────────────────────────────────────────────────────────

# Full catalogue with plain-English descriptions the LLM uses to reason
ANALYSIS_CATALOGUE = {
    # Drug analyses
    "drug_evidence":           "Phase distribution, trial statuses, enrollment, conditions, and sponsors for a specific drug",
    "drug_competition":        "Other drugs tested in the same conditions as the target drug (competitive landscape)",
    "drug_geo":                "Countries where trials for a drug are conducted (geographic footprint)",
    "drug_paths":              "Multi-hop Drug→Trial→Condition→Sponsor paths (used for GraphRAG context)",
    # Disease analyses
    "disease_landscape":       "All drugs trialled for a disease, ranked by trial count (treatment landscape)",
    "disease_phase":           "Phase distribution across trials for a disease (phase funnel/progression)",
    "disease_enrollment":      "Enrollment size distribution and median by phase for a disease",
    "disease_design":          "Trial design breakdown: allocation, blinding, arm types for a disease",
    "disease_sponsor_diversity": "Sponsor mix (industry vs academic vs government) for a disease",
    # Sponsor analyses
    "sponsor_portfolio":       "Conditions/diseases a sponsor is running trials in (condition portfolio)",
    "sponsor_drugs":           "Drugs/interventions a sponsor is testing across its trials",
    "sponsor_pipeline":        "Sponsor trial counts broken down by phase and status (pipeline)",
    "sponsor_geo":             "Countries where a sponsor is running trials (geographic reach)",
    "sponsor_collaborators":   "Other sponsors who appear in the same trials (collaborator network)",
    # Network analyses
    "centrality":              "Degree and betweenness centrality across the drug-condition graph",
    "repurposing":             "Drugs studied across the most distinct conditions (repurposing signals)",
    # Geo / temporal
    "geo_density":             "Trial counts by country globally (trial density map)",
    "trial_timeline":          "Trial start activity over time, monthly (temporal trend)",
}

# Grouped by entity type — used to constrain LLM choices
CATALOGUE_BY_TYPE = {
    "drug":    ["drug_evidence", "drug_competition", "drug_geo", "drug_paths"],
    "disease": ["disease_landscape", "disease_phase", "disease_enrollment",
                "disease_design", "disease_sponsor_diversity"],
    "sponsor": ["sponsor_portfolio", "sponsor_drugs", "sponsor_pipeline",
                "sponsor_geo", "sponsor_collaborators"],
    "network": ["centrality", "repurposing"],
    "geo":     ["geo_density", "trial_timeline"],
}


def llm_select_analyses(
    question: str,
    entity_type: str,
    entity_value: str,
    ollama_url: str,
    ollama_model: str,
    conversation_history: list | None = None,
) -> list:
    """
    Use a local LLM (via Ollama) to select which analyses to run.

    The LLM receives:
      - The user's question (+ recent conversation context for follow-ups)
      - The resolved entity type and name
      - A numbered menu of available analyses with plain-English descriptions

    It returns a JSON list of analysis keys.  The response is validated against
    ANALYSIS_CATALOGUE so hallucinated keys are silently dropped.

    Falls back to the entity-type default if the LLM call fails or returns nothing.
    """
    # Build the numbered menu — only show options relevant to this entity type
    valid_keys = CATALOGUE_BY_TYPE.get(entity_type, list(ANALYSIS_CATALOGUE.keys()))
    menu_lines = []
    for key in valid_keys:
        desc = ANALYSIS_CATALOGUE.get(key, key)
        menu_lines.append(f"  - {key}: {desc}")
    menu_str = "\n".join(menu_lines)

    # Include last 2 turns of conversation so the LLM understands follow-up context
    history_str = ""
    if conversation_history:
        recent = conversation_history[-4:]  # last 2 turns (user + assistant each)
        pairs = []
        for m in recent:
            role = "User" if m["role"] == "user" else "Assistant"
            pairs.append(f"{role}: {m['content'][:200]}")
        if pairs:
            history_str = "\n\nRecent conversation context:\n" + "\n".join(pairs)

    system = (
        "You are a clinical trials data routing system. "
        "Your only job is to select which analyses to run for a given question. "
        "You must respond with ONLY a valid JSON array of analysis key strings, "
        "nothing else — no explanation, no markdown, no preamble. "
        "Example valid response: [\"drug_evidence\", \"drug_geo\"]"
    )

    user_prompt = (
        f"Question: {question}{history_str}\n\n"
        f"Resolved entity: {entity_type.upper()} = \"{entity_value}\"\n\n"
        f"Available analyses for {entity_type}:\n{menu_str}\n\n"
        "Select the analyses that directly answer the question. "
        "Include only what is relevant — do not include everything. "
        "If the question is broad or general, include all available analyses. "
        "Respond with a JSON array of key strings only."
    )

    try:
        payload = {
            "model": ollama_model,
            "stream": False,
            "options": {"num_predict": 150, "temperature": 0},
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user_prompt},
            ],
        }
        resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()

        # Parse — strip markdown fences if present
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
        parsed = json.loads(raw)

        if not isinstance(parsed, list):
            raise ValueError("LLM did not return a list")

        # Validate — keep only keys that exist in the catalogue
        valid = [k for k in parsed if k in ANALYSIS_CATALOGUE]

        # Ensure drug_paths is always included for drug queries (needed for GraphRAG)
        if entity_type == "drug" and "drug_paths" not in valid:
            valid.append("drug_paths")

        if valid:
            return valid

    except Exception:
        pass  # fall through to defaults

    # Fallback: return all analyses for this entity type
    return CATALOGUE_BY_TYPE.get(entity_type, ["drug_evidence", "drug_paths"])


def detect_intent(
    question: str,
    kg: KGClient | None = None,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llama3",
    conversation_history: list | None = None,
) -> dict:
    """
    LLM-first intent classifier (Option A).

    Stage 1 — NCT ID short-circuit (regex, instant)
    Stage 2 — LLM entity extraction: ask the LLM what entity is named
    Stage 3 — KG confirmation: verify the entity exists in the graph
    Stage 4 — Keyword fallback: for network/geo queries with no named entity

    Returns:
      entity_type:  drug | disease | sponsor | network | geo | nct
      entity_value: canonical KG name (empty string if none)
      search_token: original LLM-extracted name (for broad CONTAINS matching)
      analyses:     None (sentinel — caller invokes llm_select_analyses)
      resolution:   "nct_id" | "llm_kg" | "llm_only" | "keyword" | "fallback"
    """
    q = question.lower()

    # ── Stage 1: NCT ID short-circuit ─────────────────────────
    nct_match = re.search(r'\b(NCT\d{8})\b', question, re.IGNORECASE)
    if nct_match:
        nct_id = nct_match.group(1).upper()
        return {
            "entity_type":  "nct",
            "entity_value": nct_id,
            "search_token": nct_id,
            "analyses":     ["nct_detail"],
            "resolution":   "nct_id",
        }

    # ── Stage 2: LLM entity extraction ────────────────────────
    entity_type   = None
    entity_value  = ""
    search_token  = ""
    resolution    = "fallback"

    llm_result = llm_extract_entity(
        question=question,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        conversation_history=conversation_history,
    )

    if llm_result:
        et = llm_result["entity_type"]
        ev = llm_result["entity_value"]

        # network/geo have no named entity — accept directly
        if et in ("network", "geo"):
            entity_type  = et
            search_token = ""
            resolution   = "llm_only"
        elif kg is not None and ev:
            # ── Stage 3: KG confirmation ───────────────────────
            confirmed = _kg_confirm(kg, ev, et)
            if confirmed:
                entity_type  = confirmed["entity_type"]
                entity_value = confirmed["entity_value"]
                search_token = confirmed["search_token"]
                resolution   = "llm_kg"
            else:
                # LLM extracted a name but it is not in the KG —
                # store what the LLM said but mark as unconfirmed
                entity_type  = et
                entity_value = ""     # empty signals "not in KG"
                search_token = ev
                resolution   = "llm_only"
        elif ev:
            # No KG available — trust LLM directly
            entity_type  = et
            entity_value = ev
            search_token = ev
            resolution   = "llm_only"

    # ── Stage 4: keyword fallback for network/geo only ────────
    if entity_type is None:
        network_kw = ["network","centrality","community","repurposing",
                      "bridge","graph","cluster","structural"]
        geo_kw     = ["country","countries","global","geography","region",
                      "timeline","trend","temporal","density","heatmap","worldwide"]
        if any(k in q for k in network_kw):
            entity_type = "network"
            resolution  = "keyword"
        elif any(k in q for k in geo_kw):
            entity_type = "geo"
            resolution  = "keyword"
        # If still None — entity_value stays "" and the caller will surface an error

    return {
        "entity_type":  entity_type or "",
        "entity_value": entity_value,
        "search_token": search_token,
        "analyses":     None,          # sentinel — caller calls llm_select_analyses
        "resolution":   resolution,
    }


def run_analyses(kg: KGClient, intent: dict, defaults: dict) -> list:
    """
    Execute all analyses implied by the intent.
    Entity value is always the KG-resolved name.
    The SEARCH TOKEN used in Cypher is the original user token (broader match)
    so that e.g. "amgen" catches all Amgen branches, not just the first resolved name.
    """
    et  = intent["entity_type"]
    ev  = intent["entity_value"]           # KG-resolved canonical name
    tok = intent.get("search_token", ev)   # original user token for broader Cypher CONTAINS

    # For each entity dimension, pick the search token when type matches,
    # otherwise fall back silently to defaults (these will not appear in UI params).
    if et == "drug":
        drug    = tok or ev or defaults["drug"]
        disease = defaults["disease"]
        sponsor = defaults["sponsor"]
    elif et == "disease":
        drug    = defaults["drug"]
        disease = tok or ev or defaults["disease"]
        sponsor = defaults["sponsor"]
    elif et == "sponsor":
        drug    = defaults["drug"]
        disease = defaults["disease"]
        sponsor = tok or ev or defaults["sponsor"]
    else:
        # network / geo — no named entity
        drug    = defaults["drug"]
        disease = defaults["disease"]
        sponsor = defaults["sponsor"]

    # The "active" param is what matters; others are irrelevant for this query
    active_param = {
        "drug":    {"drug":    drug},
        "disease": {"disease": disease},
        "sponsor": {"sponsor": sponsor},
        "network": {},
        "geo":     {},
    }.get(et, {})

    results = []

    def _add(key, title, fig, df):
        """Store result — params are always the active_param for this intent."""
        results.append({
            "key":    key,
            "title":  title,
            "fig":    fig,
            "df":     df,
            "cypher": CYPHER_REGISTRY.get(key, "— query not catalogued —"),
            "params": active_param,
        })

    for key in intent["analyses"]:
        try:
            if key == "drug_evidence":
                fig, df = drug_evidence(kg, drug)
                _add(key, f"Evidence Profile — {drug}", fig, df)

            elif key == "drug_competition":
                fig, df = drug_competition(kg, drug)
                _add(key, f"Competitive Landscape — {drug}", fig, df)

            elif key == "drug_geo":
                fig, df = drug_geo(kg, drug)
                _add(key, f"Geographic Footprint — {drug}", fig, df)

            elif key == "drug_paths":
                df = drug_paths(kg, drug)
                _add(key, f"Multi-hop Paths — {drug}", None, df)

            elif key == "disease_landscape":
                fig, df = disease_landscape(kg, disease)
                _add(key, f"Treatment Landscape — {disease}", fig, df)

            elif key == "disease_phase":
                fig, df = disease_phase_progression(kg, disease)
                _add(key, f"Phase Progression — {disease}", fig, df)

            elif key == "disease_enrollment":
                fig, df = disease_enrollment(kg, disease)
                _add(key, f"Enrollment Analysis — {disease}", fig, df)

            elif key == "disease_design":
                fig, df = disease_design(kg, disease)
                _add(key, f"Trial Design Patterns — {disease}", fig, df)

            elif key == "disease_sponsor_diversity":
                fig, df = disease_sponsor_diversity(kg, disease)
                _add(key, f"Sponsor Diversity — {disease}", fig, df)

            elif key == "sponsor_portfolio":
                fig, df = sponsor_portfolio(kg, sponsor)
                _add(key, f"Condition Portfolio — {sponsor}", fig, df)

            elif key == "sponsor_pipeline":
                fig, df = sponsor_pipeline(kg, sponsor)
                _add(key, f"Pipeline by Phase — {sponsor}", fig, df)

            elif key == "sponsor_geo":
                fig, df = sponsor_geo(kg, sponsor)
                _add(key, f"Geographic Reach — {sponsor}", fig, df)

            elif key == "sponsor_drugs":
                fig, df = sponsor_drugs(kg, sponsor)
                _add(key, f"Drug Portfolio — {sponsor}", fig, df)

            elif key == "sponsor_collaborators":
                fig, df = sponsor_collaborators(kg, sponsor)
                _add(key, f"Collaborator Network — {sponsor}", fig, df)

            elif key == "centrality":
                fig, df = centrality(kg)
                _add(key, "Drug–Condition Network Centrality", fig, df)

            elif key == "repurposing":
                fig, df = drug_repurposing(kg)
                _add(key, "Drug Repurposing Signals", fig, df)

            elif key == "geo_density":
                fig, df = trial_density(kg)
                _add(key, "Global Trial Density", fig, df)

            elif key == "trial_timeline":
                fig, df = trial_timeline(kg)
                _add(key, "Trial Activity Timeline", fig, df)

            elif key == "nct_detail":
                nct_id = intent.get("entity_value", "")
                trial  = query_nct_detail(kg, nct_id)
                if trial:
                    df  = pd.DataFrame([trial])
                    _add(key, f"Trial Detail — {nct_id}", None, df)
                else:
                    _add(key, f"Trial Detail — {nct_id}", None,
                         pd.DataFrame({"note": [f"{nct_id} not found in knowledge graph"]}))

        except Exception as exc:
            _add(key, key, None, pd.DataFrame({"error": [str(exc)]}))

    return results


def build_data_summary(results: list) -> str:
    """Produce a concise textual summary of all analysis results for the LLM."""
    lines = []
    for r in results:
        df = r["df"]
        if df is None or df.empty:
            continue
        lines.append(f"\n### {r['title']}")
        try:
            lines.append(df.head(20).to_string(index=False))
        except Exception:
            lines.append(str(df.head(20)))
    return "\n".join(lines) if lines else "No structured data available."


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Brand ──
    st.markdown("""
    <div style="padding:1.6rem 1rem 1rem;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="width:28px;height:28px;background:var(--accent);border-radius:6px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:14px;flex-shrink:0;">⬡</div>
            <div>
                <div style="font-size:0.62rem;font-weight:600;letter-spacing:0.14em;
                            text-transform:uppercase;color:var(--text-faint);">Clinical Intelligence</div>
                <div style="font-size:1.05rem;font-weight:700;color:var(--text-primary);
                            letter-spacing:-0.02em;line-height:1.1;">TrialGraph</div>
            </div>
        </div>
        <div style="font-size:0.68rem;color:var(--text-faint);
                    padding-left:36px;letter-spacing:0.02em;">
            GraphRAG · Ollama · Neo4j
        </div>
    </div>
    <div style="height:1px;background:var(--border-subtle);margin:0 1rem 0.5rem;"></div>
    """, unsafe_allow_html=True)

    # ── Card helper ─────────────────────────────────────────────
    def _card_start(label):
        st.markdown(
            f'<div style="font-size:0.6rem;font-weight:700;letter-spacing:0.16em;'
            f'text-transform:uppercase;color:var(--text-faint);'
            f'padding:1rem 0 0.45rem;">{label}</div>',
            unsafe_allow_html=True,
        )

    # ── Neo4j card ──────────────────────────────────────────────
    _card_start("Graph database")
    neo_uri  = st.text_input("URI",      value="neo4j://localhost:7687", placeholder="Bolt URI")
    neo_user = st.text_input("Username", value="neo4j",        placeholder="Username")
    neo_pass = st.text_input("Password", value="CTrail@123",   placeholder="Password", type="password")
    if st.button("Connect to Neo4j", use_container_width=True, type="primary"):
        try:
            kg = KGClient(neo_uri, neo_user, neo_pass)
            if kg.test_connection():
                st.session_state.kg = kg
                st.session_state.connected = True
                st.success("Connected.")
            else:
                st.error("Connection failed.")
        except Exception as e:
            st.error(str(e))
    if st.session_state.connected:
        st.markdown('<span class="status-pill status-ok"><span class="dot dot-ok"></span>Neo4j connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-err"><span class="dot dot-err"></span>Not connected</span>', unsafe_allow_html=True)

    # ── Ollama card ─────────────────────────────────────────────
    _card_start("Local LLM")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434", placeholder="http://localhost:11434")
    c1, c2 = st.columns([3, 2])
    with c1:
        fallback = ["llama3","llama3:8b","llama3.2","mistral","mistral:7b",
                    "gemma3","phi4","deepseek-r1:7b","qwen2.5:7b"]
        model_list = st.session_state.ollama_models or fallback
        ollama_model = st.selectbox("Model", model_list, label_visibility="collapsed")
    with c2:
        if st.button("Detect", use_container_width=True):
            models = get_ollama_models(ollama_url)
            if models:
                st.session_state.ollama_models = models
                st.success(f"{len(models)} found.")
            else:
                st.warning("Not reachable")
    running = ollama_is_running(ollama_url)
    if running:
        st.markdown('<span class="status-pill status-ok"><span class="dot dot-ok"></span>Ollama running</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-err"><span class="dot dot-err"></span>Ollama not detected</span>', unsafe_allow_html=True)

    # ── Session card ────────────────────────────────────────────
    _card_start("Session")
    st.markdown(
        '<p style="font-size:0.72rem;color:var(--text-faint);line-height:1.55;margin:0 0 0.6rem;">'
        "Entity names are resolved automatically from your question via the knowledge graph.</p>",
        unsafe_allow_html=True,
    )
    if st.button("Clear session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.analysis_panels = []
        st.session_state.conv_context = {
            "entity_type": None, "entity_value": "", "search_token": ""
        }
        st.rerun()

    st.markdown("""
    <div style="margin-top:1.5rem;padding:0.6rem 0;text-align:center;
                font-size:0.62rem;color:var(--text-faint);
                letter-spacing:0.06em;font-family:var(--font-ui);
                border-top:1px solid var(--border-subtle);">
        v1.0 &nbsp;&middot;&nbsp; TrialGraph
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  MAIN LAYOUT — two columns
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:0.8rem 0 0.9rem;border-bottom:1px solid var(--border-subtle);
            margin-bottom:1.4rem;">
    <div style="display:flex;align-items:center;gap:14px;">
        <div style="width:36px;height:36px;background:var(--accent);border-radius:8px;
                    display:flex;align-items:center;justify-content:center;font-size:18px;">⬡</div>
        <div>
            <div style="font-size:1.3rem;font-weight:700;color:var(--text-primary);
                        letter-spacing:-0.03em;line-height:1.1;">TrialGraph</div>
            <div style="font-size:0.7rem;color:var(--text-muted);letter-spacing:0.04em;
                        margin-top:2px;">Clinical trial intelligence · Knowledge Graph · GraphRAG</div>
        </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
        <div style="font-size:0.68rem;color:var(--text-faint);letter-spacing:0.06em;
                    text-transform:uppercase;">Agent interface</div>
    </div>
</div>
""", unsafe_allow_html=True)

chat_col, analysis_col = st.columns([6, 7], gap="large")


# ─────────────────────────────────────────────────────────────
#  LEFT COLUMN — CHAT AGENT
# ─────────────────────────────────────────────────────────────

with chat_col:
    st.markdown("""
    <div class="panel-header panel-header-chat">
        <span>Chat agent</span>
        <span class="panel-tag">GraphRAG · Ollama</span>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.connected:
        st.markdown("""
        <div class="empty-card">
            <div class="empty-card-title">Not connected</div>
            <div class="empty-card-body">
                Enter your Neo4j credentials in the sidebar and click
                <strong style="color:var(--accent);">Connect to Neo4j</strong> to begin.
            </div>
        </div>""", unsafe_allow_html=True)
    elif not running:
        st.markdown("""
        <div class="empty-card">
            <div class="empty-card-title">Ollama not running</div>
            <div class="empty-card-body">
                Start it with <code>ollama serve</code> then
                pull a model with <code>ollama pull llama3</code>.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Render chat history
        for i, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # Collapsible metadata panel for assistant messages
                if msg["role"] == "assistant" and "meta" in msg and msg["meta"]:
                    meta = msg["meta"]
                    with st.expander("Query Details", expanded=False):
                        resolution_label = {
                            "kg_match":             "KG lookup — matched directly against graph nodes",
                            "keyword":              "Keyword scoring — no direct KG match found",
                            "fallback":             "Keyword scoring matched a type but no name was found",
                    "unresolved":            "No entity resolved — query halted, no defaults used",
                            "conversation_context": "Conversation context — inherited from prior turn",
                        }.get(meta.get("resolution", ""), meta.get("resolution", "unknown"))

                        st.markdown(f"**Entity type:** `{meta.get('entity_type','—')}`")
                        st.markdown(f"**Entity resolved (canonical):** `{meta.get('entity_value','—') or '— using default'}`")
                        _st = meta.get("search_token","")
                        _ev = meta.get("entity_value","")
                        if _st and _st != _ev:
                            st.markdown(
                                f"**Search token (Cypher CONTAINS):** `{_st}`  "
                                f"— covers all branches/subsidiaries."
                            )
                        st.markdown(f"**Resolution method:** {resolution_label}")
                        st.markdown(f"**Analyses triggered:** `{', '.join(meta.get('analyses', []))}`")
                        st.markdown(f"**Context size:** {meta.get('context_length', 0):,} characters")
                        st.markdown(f"**Model:** `{meta.get('model','—')}`")

                        if meta.get("params"):
                            st.markdown("**Parameters passed to Cypher:**")
                            st.json(meta["params"])

                        cyphers = meta.get("cyphers", {})
                        if cyphers:
                            st.markdown("**Cypher Queries Executed:**")
                            for name, query in cyphers.items():
                                st.markdown(f"*{name}*")
                                st.code(query, language="cypher")

        # Chat input — checkbox gates whether analyses/plots are generated
        generate_plots = st.checkbox(
            "Generate plots and summaries for this query",
            value=False,
            key="cb_gen_plots",
            help="When checked, the Analysis Agent will run KG queries and render plots. "
                 "Leave unchecked for a faster text-only answer.",
        )

        if prompt := st.chat_input("Ask a clinical question — drug, disease, sponsor, network..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # ── Entity resolution ──────────────────────────────────────
            with st.status("Processing query...", expanded=True) as _status:
                st.write("Extracting entity from question...")
                intent = detect_intent(
                    question=prompt,
                    kg=st.session_state.kg,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    conversation_history=st.session_state.chat_history[-6:],
                )

            # ── Conversation context carry-forward ──────────────────────
            _pq   = prompt.lower()
            _prev = st.session_state.conv_context
            _referential = any(w in _pq for w in [
                "their", "its", "the same", "that company", "that sponsor",
                "that drug", "those", "them", "they ", "the drug",
                "the sponsor", "the company", "the disease", "the condition",
                "flagship", "pipeline", "portfolio", "mentioned",
                "previous", "above", "earlier", "last", "same",
            ])

            if not intent["entity_value"] and _prev["entity_value"]:
                intent = {
                    **intent,
                    "entity_type":  _prev["entity_type"],
                    "entity_value": _prev["entity_value"],
                    "search_token": _prev["search_token"],
                    "resolution":   "conversation_context",
                }
            elif _referential and _prev["entity_value"]:
                intent = {
                    **intent,
                    "entity_type":  _prev["entity_type"],
                    "entity_value": _prev["entity_value"],
                    "search_token": _prev["search_token"],
                    "resolution":   "conversation_context",
                }

            # ── Firewall: hard-stop if still no entity ──────────────────
            _entity_resolved = bool(intent["entity_value"])

            if not _entity_resolved:
                # No entity from KG, no conversation context — refuse to proceed
                # with defaults. Surface a clear diagnostic message instead.
                _unresolved_msg = (
                    "**Entity not resolved.**\n\n"
                    "The question did not contain a recognisable drug, disease, or sponsor "
                    "name, and there is no prior conversation context to carry forward.\n\n"
                    "Please rephrase and name the entity explicitly — for example:\n"
                    "- *What conditions is Amgen investing in?*\n"
                    "- *Show me the trial pipeline for Pfizer.*\n"
                    "- *What phase are Semaglutide trials in?*"
                )
                with st.chat_message("assistant"):
                    st.warning(_unresolved_msg)
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": _unresolved_msg,
                    "meta":    {
                        "entity_type": "—", "entity_value": "—",
                        "resolution": "unresolved", "analyses": [],
                        "context_length": 0, "model": ollama_model,
                        "cyphers": {}, "params": {},
                    },
                })
                st.rerun()

            # Entity confirmed — update conversation context
            # NCT queries do not update the persistent entity context since
            # the user is asking about a specific trial, not switching topic.
            if intent["entity_type"] != "nct":
                st.session_state.conv_context = {
                    "entity_type":  intent["entity_type"],
                    "entity_value": intent["entity_value"],
                    "search_token": intent.get("search_token", intent["entity_value"]),
                }

            # intent_defaults are now only structural placeholders — they are
            # NEVER used when entity_value is populated (which it always is here).
            intent_defaults = {
                "drug":    intent["entity_value"] if intent["entity_type"] == "drug"    else "",
                "disease": intent["entity_value"] if intent["entity_type"] == "disease" else "",
                "sponsor": intent["entity_value"] if intent["entity_type"] == "sponsor" else "",
            }

            # ── LLM-based analysis selection (Option 3) ─────────────────
            if intent["entity_type"] == "nct":
                intent["analyses"] = ["nct_detail"]
            elif intent["analyses"] is None:
                _status.write("Selecting analyses via LLM...")
                intent["analyses"] = llm_select_analyses(
                    question=prompt,
                    entity_type=intent["entity_type"],
                    entity_value=intent["entity_value"],
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    conversation_history=st.session_state.chat_history[-6:],
                )

            if generate_plots:
                _status.write("Running knowledge graph queries...")
                results = run_analyses(st.session_state.kg, intent, intent_defaults)
                turn_label = f"Turn {len(st.session_state.analysis_panels) + 1}: {prompt[:60]}"
                st.session_state.analysis_panels.append({
                    "turn_id": len(st.session_state.analysis_panels),
                    "label":   turn_label,
                    "panels":  results,
                })
            else:
                results = []

            _status.update(
                label=f"Ready · {intent['entity_type']} · {intent['entity_value'] or 'no entity'}",
                state="complete",
                expanded=False,
            )

            # Build context — route by entity type
            et = intent["entity_type"]
            ev = intent["entity_value"]
            data_summary = build_data_summary(results)

            _ev_drug    = ev if et == "drug"    and ev else intent_defaults["drug"]
            _ev_disease = ev if et == "disease" and ev else intent_defaults["disease"]
            _ev_sponsor = ev if et == "sponsor" and ev else intent_defaults["sponsor"]

            if et == "nct":
                # Build KG context directly from the trial detail record
                trial_detail = query_nct_detail(st.session_state.kg, ev)
                kg_context   = build_nct_context(trial_detail) if trial_detail else (
                    f"Trial {ev} was not found in the knowledge graph."
                )
            elif et == "disease":
                kg_context = build_graphrag_context(
                    st.session_state.kg, disease=_ev_disease, limit=150)
            elif et == "sponsor":
                kg_context = build_graphrag_context(
                    st.session_state.kg, sponsor=_ev_sponsor, limit=150)
            elif et in ("network", "geo"):
                kg_context = build_graphrag_context(
                    st.session_state.kg, drug=intent_defaults["drug"], limit=80)
            else:
                kg_context = build_graphrag_context(
                    st.session_state.kg, drug=_ev_drug, limit=150)

            # Build conversation context note for the system prompt
            _ctx_note = ""
            if _prev["entity_value"]:
                _ctx_note = (
                    "\n## Conversation Context\n"
                    f"The previous turn discussed: "
                    f"{_prev['entity_type']} = {_prev['entity_value']}. "
                    "When the user refers to 'their', 'its', 'the same', or uses "
                    "pronouns without naming a specific entity, interpret those "
                    f"references as referring to {_prev['entity_value']}.\n"
                )

            system_prompt = f"""You are a senior clinical trials intelligence analyst with deep expertise in drug development, clinical research, oncology, metabolic diseases, and pharmaceutical strategy.
{_ctx_note}
Your task is to answer the user's question in comprehensive detail, drawing exclusively on the structured knowledge graph data provided below.

Guidelines:
- Provide a thorough, well-structured analytical response. Do not truncate or summarise prematurely.
- Organise your response with clear headings and sub-sections where appropriate.
- Cite specific NCT IDs when referencing individual trials.
- Include quantitative observations (counts, percentages, rankings) derived directly from the data.
- Discuss clinical and strategic implications of the patterns you observe.
- If data is absent for a specific sub-question, state this explicitly rather than speculating.
- Use precise clinical and pharmaceutical terminology.
- Do not use bullet points excessively — prefer analytical prose for insights, with tables or lists only for enumerations.
- Maintain conversational continuity: if the user refers to a previously mentioned entity using pronouns or implicit references, answer in that context.

## Structured Knowledge Graph Evidence (Multi-hop Paths)
{kg_context}

## Aggregated Analysis Data
{data_summary}
"""

            history_for_llm = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]

            # Stream the response
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown(
                    '<div style="font-size:0.78rem;color:var(--text-faint);">'
                    'Generating answer...</div>',
                    unsafe_allow_html=True,
                )
                full_response = ""
                try:
                    for token in ollama_stream(
                        url=ollama_url,
                        model=ollama_model,
                        system=system_prompt,
                        messages=history_for_llm,
                    ):
                        full_response += token
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                except requests.exceptions.ConnectionError:
                    full_response = "Cannot reach Ollama. Ensure `ollama serve` is running."
                    placeholder.error(full_response)
                except Exception as e:
                    full_response = f"Error: {e}"
                    placeholder.error(full_response)

                # Build metadata for the collapsible panel
                # Only expose the parameter actually used — not all three defaults
                _active_param_display = {}
                if et == "drug":
                    _active_param_display = {"drug": _ev_drug}
                elif et == "disease":
                    _active_param_display = {"disease": _ev_disease}
                elif et == "sponsor":
                    _active_param_display = {
                        "sponsor (resolved canonical)": ev,
                        "sponsor (search token / broad match)": intent.get("search_token", ev),
                    }
                # network/geo have no named entity parameter

                meta = {
                    "entity_type":    et,
                    "entity_value":   ev,
                    "search_token":   intent.get("search_token", ev),
                    "resolution":     intent.get("resolution", "unknown"),
                    "analyses":       intent["analyses"],
                    "context_length": len(kg_context) + len(data_summary),
                    "model":          ollama_model,
                    "cyphers":        {k: CYPHER_REGISTRY[k] for k in intent["analyses"] if k in CYPHER_REGISTRY},
                    "params":         _active_param_display,
                }

                # Show expander inline immediately after generation
                with st.expander("Query Details", expanded=False):
                    resolution_label = {
                        "kg_match": "KG lookup — matched directly against graph nodes",
                        "keyword":  "Keyword scoring — no direct KG match found",
                        "fallback": "Fallback — entity not resolved",
                    }.get(meta["resolution"], meta["resolution"])

                    st.markdown(f"**Entity type:** `{meta['entity_type']}`")
                    st.markdown(f"**Entity resolved (canonical):** `{meta['entity_value'] or '— using default'}`")
                    if meta.get("search_token") and meta["search_token"] != meta["entity_value"]:
                        st.markdown(
                            f"**Search token (used in Cypher CONTAINS):** `{meta['search_token']}`  "
                            f"— this matches ALL entities whose name contains this string, "
                            f"covering all branches/subsidiaries."
                        )
                    st.markdown(f"**Resolution method:** {resolution_label}")
                    st.markdown(f"**Analyses triggered:** `{', '.join(meta['analyses'])}`")
                    st.markdown(f"**Context size:** {meta['context_length']:,} characters")
                    st.markdown(f"**Model:** `{meta['model']}`")
                    if meta.get("params"):
                        st.markdown("**Parameters passed to Cypher:**")
                        st.json(meta["params"])
                    st.markdown("**Cypher Queries Executed:**")
                    for name, q in meta["cyphers"].items():
                        st.markdown(f"*{name}*")
                        st.code(q, language="cypher")

            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": full_response,
                "meta":    meta,
            })

            st.rerun()


# ─────────────────────────────────────────────────────────────
#  RIGHT COLUMN — ANALYSIS AGENT (Plots + Summaries)
# ─────────────────────────────────────────────────────────────

with analysis_col:
    st.markdown("""
    <div class="panel-header panel-header-analysis">
        <span>Analysis agent</span>
        <span class="panel-tag">Plots · Summaries</span>
    </div>""", unsafe_allow_html=True)

    all_turns = st.session_state.analysis_panels

    if not all_turns:
        st.markdown("""
        <div class="empty-card">
            <div class="empty-card-title">No analyses yet</div>
            <div class="empty-card-body">
                Tick <strong style="color:var(--accent);">Generate plots and summaries</strong>
                in the chat panel before submitting a question.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        summary_system = """You are a clinical trials data analyst.
Given a table of structured data from a clinical trials knowledge graph,
write a concise but insightful analytical commentary (3-5 sentences).
Focus on the most notable pattern, outlier, or clinical implication in the data.
Be specific with numbers. Use precise language. Do not use bullet points."""

        # Render turns oldest-first (Turn 1, Turn 2, …).
        # "Data & Query" detail blocks are plain top-level expanders — no nesting.
        for turn in all_turns:
            is_latest = (turn["turn_id"] == all_turns[-1]["turn_id"])  # last = most recent

            # Turn header
            turn_num = turn["turn_id"] + 1
            turn_q   = turn["label"][len(f"Turn {turn_num}: "):]
            css_cls  = "turn-label-latest" if is_latest else "turn-label-prior"
            st.markdown(
                f'<div class="{css_cls}">'
                f'<div class="turn-num">Turn {turn_num}</div>'
                f'<div class="turn-q">{turn_q}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            panels = turn["panels"]
            if not panels:
                st.caption("No analysis panels for this turn.")
                st.markdown("---")
                continue

            for panel in panels:
                st.markdown(
                    f'<div class="chart-title">{panel["title"]}</div>',
                    unsafe_allow_html=True,
                )

                # Chart
                if panel.get("fig") is not None:
                    try:
                        _fig = panel["fig"]
                        _fig.patch.set_alpha(0)          # transparent outer background
                        for _ax in _fig.get_axes():
                            _ax.patch.set_alpha(0)       # transparent axes background
                        st.pyplot(_fig, transparent=True, use_container_width=True)
                    except Exception:
                        st.caption("Chart could not be rendered.")

                # Data + LLM summary
                df = panel.get("df")
                if df is not None and not df.empty:
                    # Generate and cache LLM summary
                    cached_summary = panel.get("summary")
                    if cached_summary is None and running:
                        try:
                            data_str = df.head(25).to_string(index=False)
                            summary_prompt = (
                                f"Analysis: {panel['title']}\n\n"
                                f"Data:\n{data_str}\n\n"
                                f"Write a concise analytical summary of what this data reveals."
                            )
                            cached_summary = ollama_complete(
                                url=ollama_url,
                                model=ollama_model,
                                system=summary_system,
                                prompt=summary_prompt,
                            )
                            panel["summary"] = cached_summary
                        except Exception:
                            cached_summary = None

                    if cached_summary:
                        st.markdown(
                            f'<div class="analysis-summary">{cached_summary}</div>',
                            unsafe_allow_html=True,
                        )

                    # Data & query detail — top-level expander, no nesting
                    with st.expander(f"Data & Query — {panel['title']}", expanded=False):
                        st.dataframe(df, use_container_width=True)
                        st.markdown("**Cypher Query:**")
                        st.code(panel.get("cypher", "—"), language="cypher")
                        if panel.get("params"):
                            st.markdown("**Parameters:**")
                            st.json(panel["params"])

                st.markdown("---")

            # Separator between turns
            st.markdown('<hr class="turn-sep">', unsafe_allow_html=True)
