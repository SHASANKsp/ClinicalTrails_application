"""
ClinicalTrials KG — Streamlit Application
==========================================
Integrates all analytics modules with an Ollama-powered GraphRAG chat agent.

Run:  streamlit run streamlit_app.py

Requirements:
  pip install streamlit neo4j networkx pandas matplotlib seaborn requests python-louvain

Ollama must be running locally:
  ollama serve
  ollama pull llama3          # or any model you prefer
"""

import streamlit as st
import requests
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicalTrials KG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0b0e18; }
  [data-testid="stSidebar"]          { background: #11151f; border-right: 1px solid #1e2535; }
  h1, h2, h3                        { color: #eaf0ff !important; }
  .stTabs [data-baseweb="tab"]       { color: #8899bb; }
  .stTabs [aria-selected="true"]     { color: #5b8af5 !important; border-bottom-color: #5b8af5 !important; }
  div[data-testid="stChatMessage"]   { background: #11151f; border: 1px solid #1e2535; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "kg"            not in st.session_state: st.session_state.kg = None
if "chat_history"  not in st.session_state: st.session_state.chat_history = []
if "connected"     not in st.session_state: st.session_state.connected = False
if "ollama_models" not in st.session_state: st.session_state.ollama_models = []


# ══════════════════════════════════════════════════════════════
#  OLLAMA HELPERS
# ══════════════════════════════════════════════════════════════

def ollama_is_running(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False

def get_ollama_models(base_url: str) -> list:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=4)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []

def ollama_chat_stream(base_url: str, model: str, system: str, messages: list):
    """Stream tokens from Ollama /api/chat. Yields str chunks."""
    payload = {
        "model": model,
        "stream": True,
        "messages": [{"role": "system", "content": system}] + messages,
    }
    with requests.post(f"{base_url}/api/chat", json=payload,
                       stream=True, timeout=180) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧬 ClinicalTrials KG")
    st.markdown("---")

    st.markdown("### 🔌 Neo4j Connection")
    neo_uri  = st.text_input("URI",      value="neo4j://localhost:7687")
    neo_user = st.text_input("User",     value="neo4j")
    neo_pass = st.text_input("Password", value="CTrail@123", type="password")

    if st.button("Connect to Neo4j", use_container_width=True, type="primary"):
        try:
            kg = KGClient(neo_uri, neo_user, neo_pass)
            if kg.test_connection():
                st.session_state.kg = kg
                st.session_state.connected = True
                st.success("✅ Connected!")
            else:
                st.error("Connection failed.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.caption("🟢 Connected" if st.session_state.connected else "🔴 Disconnected")

    st.markdown("---")
    st.markdown("### 🦙 Ollama (Local LLM)")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")

    if st.button("🔍 Detect Models", use_container_width=True):
        models = get_ollama_models(ollama_url)
        if models:
            st.session_state.ollama_models = models
            st.success(f"Found {len(models)} model(s): {', '.join(models)}")
        else:
            st.warning("Ollama not reachable. Run: `ollama serve`")

    fallback_models = [
        "llama3", "llama3:8b", "llama3.2", "mistral", "mistral:7b",
        "gemma3", "gemma3:12b", "phi3", "phi4",
        "deepseek-r1:7b", "qwen2.5:7b",
    ]
    model_list = st.session_state.ollama_models or fallback_models
    ollama_model = st.selectbox("Model", model_list, index=0)

    running = ollama_is_running(ollama_url)
    st.caption(f"Ollama: {'🟢 Running' if running else '🔴 Not detected'}")
    st.markdown("""
    <small style='color:#6a7490'>
    Install: <a href='https://ollama.com' style='color:#5b8af5'>ollama.com</a><br>
    Then: <code>ollama pull llama3</code>
    </small>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Defaults")
    default_drug    = st.text_input("Drug",    "Semaglutide")
    default_disease = st.text_input("Disease", "Diabetes")
    default_sponsor = st.text_input("Sponsor", "Novo Nordisk")
    st.caption("v1.0 · Neo4j + Ollama + Streamlit")


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def require_connection():
    if not st.session_state.connected or st.session_state.kg is None:
        st.warning("⚠️ Please connect to Neo4j first (sidebar).")
        st.stop()

def show_fig_and_data(fig, df, title=""):
    if fig is not None:
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("No data returned.")
    if df is not None and not df.empty:
        with st.expander(f"📋 Data — {title}"):
            st.dataframe(df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════

st.title("🧬 ClinicalTrials Knowledge Graph")
st.caption("Drug · Disease · Sponsor · Network · Geo · Ollama GraphRAG Agent")

tabs = st.tabs([
    "💊 Drug", "🧬 Disease", "🏢 Sponsor",
    "🕸️ Network", "🌍 Geo & Time", "🦙 Agent Chat",
])

# ── TAB 1: DRUG ───────────────────────────────────────────────
with tabs[0]:
    st.header("💊 Drug Intelligence")
    require_connection()
    drug = st.text_input("Drug name", default_drug, key="drug_input")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Evidence Profile")
        if st.button("Run Evidence Profile", key="btn_ev"):
            with st.spinner():
                show_fig_and_data(*drug_evidence(st.session_state.kg, drug), "Evidence")
        st.subheader("Geographic Footprint")
        if st.button("Run Geo Footprint", key="btn_geo"):
            with st.spinner():
                show_fig_and_data(*drug_geo(st.session_state.kg, drug), "Geo")
    with c2:
        st.subheader("Competitive Landscape")
        if st.button("Run Competition", key="btn_comp"):
            with st.spinner():
                show_fig_and_data(*drug_competition(st.session_state.kg, drug), "Competition")
        st.subheader("Multi-hop Paths")
        if st.button("Fetch Paths", key="btn_paths"):
            with st.spinner():
                df = drug_paths(st.session_state.kg, drug)
                st.dataframe(df.head(50), use_container_width=True) if not df.empty else st.info("No paths.")
    st.markdown("---")
    st.subheader("🕸️ Drug–Condition–Sponsor Subgraph")
    if st.button("Render Subgraph", key="btn_sg"):
        with st.spinner("Building network..."):
            fig, G = drug_condition_subgraph(st.session_state.kg, drug)
            if fig:
                st.pyplot(fig); plt.close(fig)
                st.caption(f"Nodes: {G.number_of_nodes()} · Edges: {G.number_of_edges()}")
            else:
                st.info("No data.")

# ── TAB 2: DISEASE ────────────────────────────────────────────
with tabs[1]:
    st.header("🧬 Disease Analytics")
    require_connection()
    disease = st.text_input("Disease / condition", default_disease, key="dis_input")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Treatment Landscape",  key="btn_ls"):  show_fig_and_data(*disease_landscape(st.session_state.kg, disease), "Landscape")
        if st.button("Phase Progression",    key="btn_ph"):  show_fig_and_data(*disease_phase_progression(st.session_state.kg, disease), "Phase")
        if st.button("Sponsor Diversity",    key="btn_sd"):  show_fig_and_data(*disease_sponsor_diversity(st.session_state.kg, disease), "Diversity")
    with c2:
        if st.button("Trial Design Patterns",key="btn_dp"):  show_fig_and_data(*disease_design(st.session_state.kg, disease), "Design")
        if st.button("Enrollment Analysis",  key="btn_en"):  show_fig_and_data(*disease_enrollment(st.session_state.kg, disease), "Enrollment")

# ── TAB 3: SPONSOR ────────────────────────────────────────────
with tabs[2]:
    st.header("🏢 Sponsor Intelligence")
    require_connection()
    sponsor = st.text_input("Sponsor name", default_sponsor, key="sp_input")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Condition Portfolio",      key="btn_po"): show_fig_and_data(*sponsor_portfolio(st.session_state.kg, sponsor), "Portfolio")
        if st.button("Geographic Reach",         key="btn_sg2"): show_fig_and_data(*sponsor_geo(st.session_state.kg, sponsor), "Geo")
        if st.button("Drug Pipeline",            key="btn_sdg"): show_fig_and_data(*sponsor_drugs(st.session_state.kg, sponsor), "Drugs")
    with c2:
        if st.button("Pipeline (Phase×Status)",  key="btn_spp"): show_fig_and_data(*sponsor_pipeline(st.session_state.kg, sponsor), "Pipeline")
        if st.button("Collaborator Network",     key="btn_col"): show_fig_and_data(*sponsor_collaborators(st.session_state.kg, sponsor), "Collaborators")
    st.markdown("---")
    st.subheader("🕸️ Sponsor Collaboration Network")
    min_sh = st.slider("Min shared trials", 1, 20, 3, key="sl_sh")
    if st.button("Render Sponsor Network", key="btn_sn"):
        with st.spinner():
            fig, G = sponsor_network(st.session_state.kg, min_shared_trials=min_sh)
            if fig: st.pyplot(fig); plt.close(fig)
            else: st.info("No data.")

# ── TAB 4: NETWORK ────────────────────────────────────────────
with tabs[3]:
    st.header("🕸️ Network & Graph Analytics")
    require_connection()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Centrality Analysis",      key="btn_ca"):  show_fig_and_data(*centrality(st.session_state.kg), "Centrality")
        if st.button("Drug Repurposing Signals", key="btn_rp"):  show_fig_and_data(*drug_repurposing(st.session_state.kg), "Repurposing")
        if st.button("Degree Distribution",      key="btn_dd"):  show_fig_and_data(*degree_distribution(st.session_state.kg), "Degree Dist")
    with c2:
        if st.button("Community Detection",      key="btn_cd"):  show_fig_and_data(*community_detection(st.session_state.kg), "Communities")
        if st.button("Bridge Drugs",             key="btn_bd"):  show_fig_and_data(*bridge_drugs(st.session_state.kg), "Bridges")
        if st.button("Graph Metrics Summary",    key="btn_gm"):
            df = graph_metrics(st.session_state.kg); st.dataframe(df, use_container_width=True)
    st.markdown("---")
    st.subheader("🌐 Condition Similarity Network")
    min_sc = st.slider("Min shared drugs", 1, 10, 2, key="sl_sc")
    if st.button("Render Condition Network", key="btn_cn"):
        with st.spinner():
            fig, G = condition_similarity_network(st.session_state.kg, min_shared=min_sc)
            if fig: st.pyplot(fig); plt.close(fig)
            else: st.info("No data.")

# ── TAB 5: GEO & TIME ─────────────────────────────────────────
with tabs[4]:
    st.header("🌍 Geo & Temporal Analytics")
    require_connection()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Global Trial Density",   key="btn_gd"): show_fig_and_data(*trial_density(st.session_state.kg), "Density")
    with c2:
        if st.button("Trial Timeline",         key="btn_tl"): show_fig_and_data(*trial_timeline(st.session_state.kg), "Timeline")
    if st.button("Country × Phase Heatmap",    key="btn_hm"):
        show_fig_and_data(*geo_phase_heatmap(st.session_state.kg), "Heatmap")

# ── TAB 6: OLLAMA AGENT ───────────────────────────────────────
with tabs[5]:
    st.header("🦙 GraphRAG Agent — Powered by Ollama")
    st.caption("The KG subgraph is fetched and injected as context before your local LLM answers. No API key needed.")

    require_connection()

    if not ollama_is_running(ollama_url):
        st.error(
            "**Ollama is not running.**  \n"
            "Open a terminal and run:  \n"
            "```\nollama serve\n```\n"
            "Then pull a model:  \n"
            "```\nollama pull llama3\n```"
        )
        st.stop()

    # ── Context selector ──────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        ctx_type = st.selectbox("Context type", ["Drug", "Disease", "Sponsor"], key="ctx_t")
    with c2:
        ctx_value = st.text_input(
            "Entity",
            value={"Drug": default_drug, "Disease": default_disease,
                   "Sponsor": default_sponsor}[ctx_type],
            key="ctx_v",
        )
    with c3:
        ctx_limit = st.slider("Max KG paths", 20, 200, 80, step=20, key="ctx_lim")

    if st.button("🔄 Clear Chat", key="btn_clr"):
        st.session_state.chat_history = []
        st.rerun()

    with st.expander("🔍 Preview KG Context injected into the LLM"):
        ctx_text = build_graphrag_context(
            st.session_state.kg,
            **{ctx_type.lower(): ctx_value, "limit": ctx_limit}
        )
        st.code(ctx_text[:3000] + ("..." if len(ctx_text) > 3000 else ""), language="text")

    st.markdown("---")

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a clinical question…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        kg_context = build_graphrag_context(
            st.session_state.kg,
            **{ctx_type.lower(): ctx_value, "limit": ctx_limit}
        )

        system_prompt = f"""You are a clinical trials intelligence analyst.
Answer questions using ONLY the knowledge graph evidence below.
Do not guess — if something is not in the evidence, say so.
Cite NCT IDs when referencing specific trials.

## Knowledge Graph Evidence
{kg_context}
"""
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
        ]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                for token in ollama_chat_stream(
                    base_url=ollama_url,
                    model=ollama_model,
                    system=system_prompt,
                    messages=messages,
                ):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except requests.exceptions.ConnectionError:
                full_response = "❌ Cannot reach Ollama. Make sure `ollama serve` is running."
                placeholder.error(full_response)
            except Exception as e:
                full_response = f"❌ Error: {e}"
                placeholder.error(full_response)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response}
        )
