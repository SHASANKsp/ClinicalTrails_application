import streamlit as st
import os
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_ollama import OllamaLLM
from langchain_community.graphs import Neo4jGraph
from repurposing_engine import RepurposingSignalEngine
from portfolio_metrics import ClinicalPortfolioMetrics

st.set_page_config(page_title="Clinical Portfolio Intelligence", layout="wide")
st.title("Clinical Portfolio Intelligence Dashboard")

# -----------------------------
# Initialize Neo4j
# -----------------------------

@st.cache_resource
def init_graph():
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "CTrail@123")
    )

graph = init_graph()
engine = RepurposingSignalEngine(graph)

# -----------------------------
# Initialize Ollama
# -----------------------------

model_name = st.sidebar.selectbox(
    "Select LLM",
    ["phi3:latest", "llama3.1:latest", "deepseek-r1:8b"]
)

@st.cache_resource
def init_llm(model):
    return OllamaLLM(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=model
    )

llm = init_llm(model_name)

# -----------------------------
# Drug Input
# -----------------------------

drug_input = st.text_input("Enter Drug Name (Exact Match)")

if drug_input:

    df = engine.fetch_data(drug_input)

    if df is None or df.empty:
        st.error("No data found.")
    else:

        metrics = ClinicalPortfolioMetrics(df)

        # -------------------------
        # Display Metrics
        # -------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Trials", metrics.total_trials())
        col2.metric("Condition Diversity", metrics.condition_diversity())
        col3.metric("Sponsor Diversity", metrics.sponsor_diversity())
        col4.metric("Geographic Spread", metrics.geographic_spread())

        col5, col6, col7 = st.columns(3)

        col5.metric("Completion Ratio", round(metrics.completion_ratio(), 2))
        col6.metric("Phase Maturity", round(metrics.phase_maturity_score(), 2))
        col7.metric("Repurposing Strength", round(metrics.repurposing_strength(), 3))

        # -------------------------
        # Heatmap
        # -------------------------

        st.subheader("Condition–Phase Heatmap")

        heatmap_data = (
            df.groupby(["condition", "phase"])["nct_id"]
            .nunique()
            .unstack(fill_value=0)
        )

        if not heatmap_data.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )
            st.pyplot(fig)

        # -------------------------
        # LLM Strategic Summary
        # -------------------------

        st.subheader("Strategic Portfolio Summary")

        summary_prompt = f"""
You are a clinical development strategist and competitive intelligence analyst.

Drug: {drug_input}

Portfolio Metrics:
Total Trials: {metrics.total_trials()}
Condition Diversity: {metrics.condition_diversity()}
Sponsor Diversity: {metrics.sponsor_diversity()}
Geographic Spread: {metrics.geographic_spread()}
Completion Ratio: {metrics.completion_ratio()}
Phase Maturity Score: {metrics.phase_maturity_score()}
Repurposing Strength: {metrics.repurposing_strength()}

Provide a scientific portfolio interpretation:
- Is this exploratory or mature repurposing?
- Is there phase progression?
- Is sponsor landscape concentrated?
- What risks are visible?
"""

        try:
            summary = llm.invoke(summary_prompt)
            st.markdown(summary)
        except Exception as e:
            st.error("LLM execution failed.")
            st.exception(e)