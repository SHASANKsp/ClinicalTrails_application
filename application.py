"""
Clinical Trials Knowledge Graph — Analytics Applications
=========================================================
  1. Drug Intelligence
  2. Disease Analytics
  3. Sponsor Intelligence
  4. Network & Graph Analytics
  5. Geo & Temporal Analytics

Refactored for Streamlit: all plot functions return (fig, df) instead of
saving to disk, so Streamlit can render them inline.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

from neo4j import GraphDatabase


# ══════════════════════════════════════════════════════════════
#  NEO4J CLIENT
# ══════════════════════════════════════════════════════════════

class KGClient:
    """Thin wrapper around the Neo4j driver."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def query(self, cypher: str, params: dict = {}) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(cypher, params)
            return [dict(r) for r in result]

    def close(self):
        self.driver.close()

    def test_connection(self) -> bool:
        try:
            self.query("RETURN 1")
            return True
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB STYLE HELPERS
# ══════════════════════════════════════════════════════════════

PALETTE = {
    "steelblue":    "#4682B4",
    "darkcyan":     "#008B8B",
    "tomato":       "#FF6347",
    "mediumseagreen": "#3CB371",
    "mediumpurple": "#9370DB",
    "slateblue":    "#6A5ACD",
    "goldenrod":    "#DAA520",
    "orange":       "#FFA500",
    "darkorange":   "#FF8C00",
    "peru":         "#CD853F",
    "purple":       "#800080",
    "indigo":       "#4B0082",
    "darkorchid":   "#9932CC",
    "teal":         "#008080",
    "skyblue":      "#87CEEB",
    "violet":       "#EE82EE",
}

def _style():
    plt.rcParams.update({
        "figure.facecolor": "#0F1117",
        "axes.facecolor":   "#0F1117",
        "axes.edgecolor":   "#333344",
        "axes.labelcolor":  "#CCCCDD",
        "text.color":       "#CCCCDD",
        "xtick.color":      "#CCCCDD",
        "ytick.color":      "#CCCCDD",
        "grid.color":       "#1E1E2E",
        "grid.linestyle":   "--",
        "axes.grid":        True,
        "axes.titlecolor":  "#EEEEFF",
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
    })

_style()


# ══════════════════════════════════════════════════════════════
#  CATEGORY 1 — DRUG INTELLIGENCE
# ══════════════════════════════════════════════════════════════

def drug_evidence(kg: KGClient, drug: str):
    """
    Full evidence profile for a drug: trials, phases, conditions, sponsors, countries.
    Returns (fig, df).
    """
    query = """
    MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)
    WHERE toLower(i.name) CONTAINS toLower($drug)
    OPTIONAL MATCH (st)-[:STUDIES]->(c:Condition)
    OPTIONAL MATCH (st)<-[:SPONSORS]-(s:Sponsor)
    OPTIONAL MATCH (st)-[:CONDUCTED_AT]->(l:Location)
    RETURN st.nct_id        AS trial,
           st.phases         AS phase,
           st.overall_status AS status,
           st.enrollment      AS enrollment,
           c.name             AS condition,
           s.name             AS sponsor,
           l.country          AS country
    """
    df = pd.DataFrame(kg.query(query, {"drug": drug}))
    if df.empty:
        return None, df

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0F1117")

    phase_counts = df["phase"].value_counts()
    phase_counts.plot.bar(ax=axes[0], color=PALETTE["steelblue"], edgecolor="none")
    axes[0].set_title(f"{drug} — Phase Distribution")
    axes[0].set_xlabel("Phase")
    axes[0].set_ylabel("Trials")
    axes[0].tick_params(axis="x", rotation=45)

    status_counts = df["status"].value_counts().head(8)
    status_counts.plot.barh(ax=axes[1], color=PALETTE["darkcyan"], edgecolor="none")
    axes[1].set_title(f"{drug} — Trial Status")
    axes[1].set_xlabel("Count")

    plt.tight_layout()
    return fig, df


def drug_competition(kg: KGClient, drug: str):
    """
    Competing drugs in the same conditions as the target drug.
    Returns (fig, df).
    """
    query = """
    MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)-[:STUDIES]->(c:Condition)
    MATCH (c)<-[:STUDIES]-(st2:Study)-[:USES_INTERVENTION]->(i2:Intervention)
    WHERE toLower(i.name)  CONTAINS toLower($drug)
      AND toLower(i2.name) <> toLower($drug)
    RETURN i2.name AS competitor, COUNT(DISTINCT st2) AS trials
    ORDER BY trials DESC
    LIMIT 30
    """
    df = pd.DataFrame(kg.query(query, {"drug": drug}))
    if df.empty:
        return None, df

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0F1117")
    df.head(15).plot.bar(x="competitor", y="trials", legend=False,
                         color=PALETTE["tomato"], edgecolor="none", ax=ax)
    ax.set_title(f"Top Competitors to {drug} (shared conditions)")
    ax.set_xlabel("Drug")
    ax.set_ylabel("Trials")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, df


def drug_geo(kg: KGClient, drug: str):
    """
    Countries conducting trials for a drug.
    Returns (fig, df).
    """
    query = """
    MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)-[:CONDUCTED_AT]->(l:Location)
    WHERE toLower(i.name) CONTAINS toLower($drug)
    RETURN l.country AS country, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    """
    df = pd.DataFrame(kg.query(query, {"drug": drug}))
    if df.empty:
        return None, df

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0F1117")
    df.head(20).plot.bar(x="country", y="trials", legend=False,
                         color=PALETTE["mediumseagreen"], edgecolor="none", ax=ax)
    ax.set_title(f"{drug} — Geographic Footprint")
    ax.set_xlabel("Country")
    ax.set_ylabel("Trials")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, df


def drug_paths(kg: KGClient, drug: str) -> pd.DataFrame:
    """
    Multi-hop paths: Drug → Study → Condition + Sponsor.
    Used as structured context for the GraphRAG pipeline.
    Returns df only (no chart).
    """
    query = """
    MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)
    WHERE toLower(i.name) CONTAINS toLower($drug)
    MATCH (st)-[:STUDIES]->(c:Condition)
    MATCH (st)<-[:SPONSORS]-(s:Sponsor)
    RETURN i.name    AS drug,
           st.nct_id  AS trial,
           c.name     AS condition,
           s.name     AS sponsor,
           s.class    AS sponsor_class
    """
    df = pd.DataFrame(kg.query(query, {"drug": drug}))
    return df


# ══════════════════════════════════════════════════════════════
#  CATEGORY 2 — DISEASE ANALYTICS
# ══════════════════════════════════════════════════════════════

def disease_landscape(kg: KGClient, disease: str):
    """
    All drugs trialled for a disease, ranked by trial count.
    Returns (fig, df).
    """
    query = """
    MATCH (st:Study)-[:STUDIES]->(c:Condition)
    WHERE toLower(c.name) CONTAINS toLower($disease)
    MATCH (st)-[:USES_INTERVENTION]->(i:Intervention)
    RETURN i.name AS drug, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    LIMIT 30
    """
    df = pd.DataFrame(kg.query(query, {"disease": disease}))
    if df.empty:
        return None, df

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F1117")
    df.head(15).plot.barh(x="drug", y="trials", legend=False,
                          color=PALETTE["slateblue"], edgecolor="none", ax=ax)
    ax.set_title(f"{disease} — Treatment Landscape (Top Drugs)")
    ax.set_xlabel("Trials")
    plt.tight_layout()
    return fig, df


def disease_design(kg: KGClient, disease: str):
    """
    Trial design breakdown: arm types, allocation, masking.
    Returns (fig, df).
    """
    query = """
    MATCH (st:Study)-[:STUDIES]->(c:Condition)
    WHERE toLower(c.name) CONTAINS toLower($disease)
    OPTIONAL MATCH (st)-[:HAS_ARM]->(a:Arm)
    RETURN st.allocation AS allocation,
           st.masking     AS masking,
           st.study_type  AS study_type,
           a.type         AS arm_type,
           COUNT(*)       AS count
    """
    df = pd.DataFrame(kg.query(query, {"disease": disease}))
    if df.empty:
        return None, df

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0F1117")
    for ax, col in zip(axes, ["allocation", "masking", "arm_type"]):
        counts = df.groupby(col)["count"].sum().sort_values(ascending=False).head(8)
        counts.plot.bar(ax=ax, color=PALETTE["mediumpurple"], edgecolor="none")
        ax.set_title(col.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(f"{disease} — Trial Design Patterns", color="#EEEEFF", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, df


def disease_phase_progression(kg: KGClient, disease: str):
    """
    Phase distribution across all trials for a disease.
    Returns (fig, df).
    """
    query = """
    MATCH (st:Study)-[:STUDIES]->(c:Condition)
    WHERE toLower(c.name) CONTAINS toLower($disease)
    RETURN st.phases AS phase, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY phase
    """
    df = pd.DataFrame(kg.query(query, {"disease": disease}))
    if df.empty:
        return None, df

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0F1117")
    df.plot.bar(x="phase", y="trials", legend=False,
                color=PALETTE["mediumpurple"], edgecolor="none", ax=ax)
    ax.set_title(f"{disease} — Phase Distribution")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Trials")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, df


def disease_enrollment(kg: KGClient, disease: str):
    """
    Enrollment size distribution for a disease.
    Returns (fig, df).
    """
    query = """
    MATCH (st:Study)-[:STUDIES]->(c:Condition)
    WHERE toLower(c.name) CONTAINS toLower($disease)
      AND st.enrollment IS NOT NULL
    RETURN st.nct_id    AS trial,
           st.enrollment AS enrollment,
           st.phases     AS phase
    """
    df = pd.DataFrame(kg.query(query, {"disease": disease}))
    if df.empty:
        return None, df

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df = df.dropna(subset=["enrollment"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0F1117")

    df["enrollment"].clip(upper=df["enrollment"].quantile(0.95)).hist(
        bins=30, ax=axes[0], color=PALETTE["steelblue"], edgecolor="#0F1117"
    )
    axes[0].set_title(f"{disease} — Enrollment Size (≤95th pct)")
    axes[0].set_xlabel("Participants")

    phase_enroll = df.groupby("phase")["enrollment"].median().sort_values()
    phase_enroll.plot.barh(ax=axes[1], color=PALETTE["steelblue"], edgecolor="none")
    axes[1].set_title("Median Enrollment by Phase")
    axes[1].set_xlabel("Median Enrollment")

    plt.suptitle(f"{disease} — Enrollment Analysis", color="#EEEEFF", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, df


def disease_sponsor_diversity(kg: KGClient, disease: str):
    """
    Industry vs Academic vs Government sponsor split for a disease.
    Returns (fig, df).
    """
    query = """
    MATCH (st:Study)-[:STUDIES]->(c:Condition)
    WHERE toLower(c.name) CONTAINS toLower($disease)
    MATCH (st)<-[:SPONSORS]-(s:Sponsor)
    RETURN s.name  AS sponsor,
           s.class AS sponsor_class,
           COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    """
    df = pd.DataFrame(kg.query(query, {"disease": disease}))
    if df.empty:
        return None, df

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#0F1117")

    df.head(12).plot.barh(x="sponsor", y="trials", ax=axes[0],
                          legend=False, color=PALETTE["darkcyan"], edgecolor="none")
    axes[0].set_title(f"Top Sponsors — {disease}")

    class_counts = df.groupby("sponsor_class")["trials"].sum().sort_values()
    class_counts.plot.barh(ax=axes[1], color=PALETTE["teal"], edgecolor="none")
    axes[1].set_title("By Sponsor Class")

    plt.suptitle(f"{disease} — Sponsor Diversity", color="#EEEEFF", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, df


# ══════════════════════════════════════════════════════════════
#  CATEGORY 3 — SPONSOR INTELLIGENCE
# ══════════════════════════════════════════════════════════════

def sponsor_portfolio(kg: KGClient, sponsor: str):
    """Condition portfolio of a sponsor. Returns (fig, df)."""
    query = """
    MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
    WHERE toLower(s.name) CONTAINS toLower($sponsor)
    MATCH (st)-[:STUDIES]->(c:Condition)
    RETURN c.name AS condition, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    LIMIT 30
    """
    df = pd.DataFrame(kg.query(query, {"sponsor": sponsor}))
    if df.empty:
        return None, df

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F1117")
    df.head(15).plot.barh(x="condition", y="trials", legend=False,
                          color=PALETTE["goldenrod"], edgecolor="none", ax=ax)
    ax.set_title(f"{sponsor} — Condition Portfolio")
    ax.set_xlabel("Trials")
    plt.tight_layout()
    return fig, df


def sponsor_geo(kg: KGClient, sponsor: str):
    """Geographic distribution of a sponsor's trials. Returns (fig, df)."""
    query = """
    MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
    WHERE toLower(s.name) CONTAINS toLower($sponsor)
    MATCH (st)-[:CONDUCTED_AT]->(l:Location)
    RETURN l.country AS country, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    """
    df = pd.DataFrame(kg.query(query, {"sponsor": sponsor}))
    if df.empty:
        return None, df

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0F1117")
    df.head(20).plot.bar(x="country", y="trials", legend=False,
                         color=PALETTE["orange"], edgecolor="none", ax=ax)
    ax.set_title(f"{sponsor} — Geographic Reach")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, df


def sponsor_pipeline(kg: KGClient, sponsor: str):
    """Phase breakdown of a sponsor's pipeline. Returns (fig, df)."""
    query = """
    MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
    WHERE toLower(s.name) CONTAINS toLower($sponsor)
    RETURN st.phases        AS phase,
           st.overall_status AS status,
           COUNT(DISTINCT st.nct_id) AS trials
    """
    df = pd.DataFrame(kg.query(query, {"sponsor": sponsor}))
    if df.empty:
        return None, df

    pivot = df.groupby(["phase", "status"])["trials"].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0F1117")
    pivot.plot.bar(stacked=True, colormap="tab10", ax=ax)
    ax.set_title(f"{sponsor} — Pipeline by Phase & Status")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Trials")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig, df


def sponsor_drugs(kg: KGClient, sponsor: str):
    """All interventions a sponsor is running trials for. Returns (fig, df)."""
    query = """
    MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
    WHERE toLower(s.name) CONTAINS toLower($sponsor)
    MATCH (st)-[:USES_INTERVENTION]->(i:Intervention)
    RETURN i.name AS drug, i.type AS intervention_type,
           COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    LIMIT 30
    """
    df = pd.DataFrame(kg.query(query, {"sponsor": sponsor}))
    if df.empty:
        return None, df

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F1117")
    df.head(15).plot.barh(x="drug", y="trials", legend=False,
                          color=PALETTE["darkorange"], edgecolor="none", ax=ax)
    ax.set_title(f"{sponsor} — Drug Pipeline")
    ax.set_xlabel("Trials")
    plt.tight_layout()
    return fig, df


def sponsor_collaborators(kg: KGClient, sponsor: str):
    """Sponsors who appear in the same trials as the target sponsor. Returns (fig, df)."""
    query = """
    MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
    WHERE toLower(s.name) CONTAINS toLower($sponsor)
    MATCH (st)<-[:SPONSORS]-(s2:Sponsor)
    WHERE toLower(s2.name) <> toLower($sponsor)
    RETURN s2.name AS collaborator, COUNT(DISTINCT st.nct_id) AS shared_trials
    ORDER BY shared_trials DESC
    LIMIT 25
    """
    df = pd.DataFrame(kg.query(query, {"sponsor": sponsor}))
    if df.empty:
        return None, df

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0F1117")
    df.head(15).plot.barh(x="collaborator", y="shared_trials",
                           legend=False, color=PALETTE["peru"], edgecolor="none", ax=ax)
    ax.set_title(f"{sponsor} — Frequent Collaborators")
    ax.set_xlabel("Shared Trials")
    plt.tight_layout()
    return fig, df


# ══════════════════════════════════════════════════════════════
#  CATEGORY 4 — NETWORK & GRAPH ANALYTICS
# ══════════════════════════════════════════════════════════════

def centrality(kg: KGClient):
    """
    Degree & betweenness centrality on the drug–condition bipartite graph.
    Returns (fig, df).
    """
    query = """
    MATCH (st:Study)-[:USES_INTERVENTION]->(i:Intervention)
    MATCH (st)-[:STUDIES]->(c:Condition)
    RETURN i.name AS drug, c.name AS condition
    """
    data = kg.query(query)
    G = nx.Graph()
    for r in data:
        if r["drug"] and r["condition"]:
            G.add_edge(r["drug"], r["condition"])

    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G, k=min(200, len(G)))
    df = pd.DataFrame({
        "node": list(deg.keys()),
        "degree_centrality":      [deg[n] for n in deg],
        "betweenness_centrality": [bet.get(n, 0) for n in deg],
    }).sort_values("degree_centrality", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F1117")
    df.head(20).plot.barh(x="node", y="degree_centrality",
                           ax=axes[0], legend=False, color=PALETTE["purple"], edgecolor="none")
    axes[0].set_title("Top 20 — Degree Centrality")

    df.sort_values("betweenness_centrality", ascending=False).head(20).plot.barh(
        x="node", y="betweenness_centrality",
        ax=axes[1], legend=False, color=PALETTE["indigo"], edgecolor="none"
    )
    axes[1].set_title("Top 20 — Betweenness Centrality")

    plt.suptitle("Drug–Condition Network Centrality", color="#EEEEFF", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, df


def community_detection(kg: KGClient):
    """
    Louvain community detection on the drug–condition graph.
    Returns (fig, df). Requires python-louvain.
    """
    if not LOUVAIN_AVAILABLE:
        return None, pd.DataFrame({"error": ["python-louvain not installed"]})

    query = """
    MATCH (st:Study)-[:USES_INTERVENTION]->(i:Intervention)
    MATCH (st)-[:STUDIES]->(c:Condition)
    RETURN i.name AS drug, c.name AS condition
    """
    data = kg.query(query)
    G = nx.Graph()
    for r in data:
        if r["drug"] and r["condition"]:
            G.add_edge(r["drug"], r["condition"])

    partition = community_louvain.best_partition(G)
    df = pd.DataFrame(partition.items(), columns=["node", "community"])
    community_sizes = df["community"].value_counts().reset_index()
    community_sizes.columns = ["community", "members"]

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0F1117")
    community_sizes.head(20).plot.bar(x="community", y="members",
                                       legend=False, color=PALETTE["slateblue"], edgecolor="none", ax=ax)
    ax.set_title("Community Sizes (Louvain — Drug–Condition Graph)")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Nodes")
    plt.tight_layout()
    return fig, df


def drug_repurposing(kg: KGClient):
    """
    Drugs studied across the largest number of distinct conditions.
    Returns (fig, df).
    """
    query = """
    MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)-[:STUDIES]->(c:Condition)
    RETURN i.name AS drug,
           COUNT(DISTINCT c.name)    AS n_conditions,
           COUNT(DISTINCT st.nct_id) AS n_trials,
           COLLECT(DISTINCT c.name)[0..8] AS sample_conditions
    ORDER BY n_conditions DESC
    LIMIT 40
    """
    df = pd.DataFrame(kg.query(query))
    if df.empty:
        return None, df

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.scatter(df["n_trials"], df["n_conditions"], alpha=0.7, color=PALETTE["darkorchid"], s=60)
    for _, row in df.head(15).iterrows():
        ax.annotate(row["drug"], (row["n_trials"], row["n_conditions"]),
                    fontsize=7, alpha=0.8, color="#CCCCDD")
    ax.set_xlabel("Number of Trials")
    ax.set_ylabel("Number of Distinct Conditions")
    ax.set_title("Drug Repurposing Signals (Breadth of Conditions Studied)")
    plt.tight_layout()
    return fig, df


# ══════════════════════════════════════════════════════════════
#  CATEGORY 5 — GEO & TEMPORAL ANALYTICS
# ══════════════════════════════════════════════════════════════

def trial_density(kg: KGClient):
    """Global trial density by country. Returns (fig, df)."""
    query = """
    MATCH (st:Study)-[:CONDUCTED_AT]->(l:Location)
    RETURN l.country AS country, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    """
    df = pd.DataFrame(kg.query(query))
    if df.empty:
        return None, df

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#0F1117")
    df.head(25).plot.bar(x="country", y="trials", legend=False,
                         color=PALETTE["teal"], edgecolor="none", ax=ax)
    ax.set_title("Global Trial Density by Country")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, df


def trial_timeline(kg: KGClient):
    """
    Trial start date distribution over time (monthly).
    Returns (fig, df).
    """
    query = """
    MATCH (st:Study)
    WHERE st.start_date IS NOT NULL
    RETURN st.start_date    AS start_date,
           st.phases         AS phase,
           st.overall_status AS status,
           COUNT(*)          AS trials
    """
    df = pd.DataFrame(kg.query(query))
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["start_date"])
    if df.empty:
        return None, df

    df["ym"] = df["start_date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("ym")["trials"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#0F1117")
    ax.plot(monthly["ym"], monthly["trials"], linewidth=1.5, color=PALETTE["steelblue"])
    ax.fill_between(monthly["ym"], monthly["trials"], alpha=0.2, color=PALETTE["steelblue"])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Trial Starts Over Time (Monthly)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trials Started")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, df


def geo_phase_heatmap(kg: KGClient):
    """
    Country × phase trial counts heatmap.
    Returns (fig, df).
    """
    query = """
    MATCH (st:Study)-[:CONDUCTED_AT]->(l:Location)
    WHERE l.country IS NOT NULL AND st.phases IS NOT NULL
    RETURN l.country AS country, st.phases AS phase,
           COUNT(DISTINCT st.nct_id) AS trials
    """
    df = pd.DataFrame(kg.query(query))
    if df.empty:
        return None, df

    pivot = (
        df.groupby(["country", "phase"])["trials"]
          .sum()
          .unstack(fill_value=0)
    )
    top_countries = pivot.sum(axis=1).nlargest(20).index
    pivot = pivot.loc[top_countries]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#0F1117")
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.3, cbar_kws={"label": "Trials"}, ax=ax)
    ax.set_title("Country × Phase Heatmap (Top 20 Countries)")
    plt.tight_layout()
    return fig, df


# ══════════════════════════════════════════════════════════════
#  GRAPHRAG — CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════

def build_graphrag_context(kg: KGClient, drug: str | None = None,
                           disease: str | None = None,
                           sponsor: str | None = None,
                           limit: int = 50) -> str:
    """
    Build a structured text context chunk from the KG for LLM grounding.
    Assembles Drug→Study→Condition→Sponsor multi-hop paths.
    """
    if drug:
        df = drug_paths(kg, drug)
        df = df.head(limit)
        if df.empty:
            return f"No data found for drug: {drug}"
        lines = [f"## Clinical Trial Evidence: {drug}\n"]
        for _, row in df.iterrows():
            lines.append(
                f"- Trial {row.get('trial', 'N/A')}: "
                f"Condition={row.get('condition', 'N/A')}, "
                f"Sponsor={row.get('sponsor', 'N/A')} ({row.get('sponsor_class', 'N/A')})"
            )
        return "\n".join(lines)

    if disease:
        query = """
        MATCH (st:Study)-[:STUDIES]->(c:Condition)
        WHERE toLower(c.name) CONTAINS toLower($disease)
        MATCH (st)-[:USES_INTERVENTION]->(i:Intervention)
        OPTIONAL MATCH (st)<-[:SPONSORS]-(s:Sponsor)
        RETURN st.nct_id AS trial, i.name AS drug,
               c.name AS condition, st.phases AS phase,
               st.overall_status AS status, s.name AS sponsor
        LIMIT $limit
        """
        rows = kg.query(query, {"disease": disease, "limit": limit})
        if not rows:
            return f"No data found for disease: {disease}"
        lines = [f"## Clinical Trial Evidence: {disease}\n"]
        for row in rows:
            lines.append(
                f"- Trial {row.get('trial', 'N/A')}: "
                f"Drug={row.get('drug', 'N/A')}, "
                f"Phase={row.get('phase', 'N/A')}, "
                f"Status={row.get('status', 'N/A')}, "
                f"Sponsor={row.get('sponsor', 'N/A')}"
            )
        return "\n".join(lines)

    if sponsor:
        query = """
        MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
        WHERE toLower(s.name) CONTAINS toLower($sponsor)
        MATCH (st)-[:STUDIES]->(c:Condition)
        MATCH (st)-[:USES_INTERVENTION]->(i:Intervention)
        RETURN st.nct_id AS trial, i.name AS drug,
               c.name AS condition, st.phases AS phase,
               st.overall_status AS status
        LIMIT $limit
        """
        rows = kg.query(query, {"sponsor": sponsor, "limit": limit})
        if not rows:
            return f"No data found for sponsor: {sponsor}"
        lines = [f"## Clinical Trial Evidence: {sponsor}\n"]
        for row in rows:
            lines.append(
                f"- Trial {row.get('trial', 'N/A')}: "
                f"Drug={row.get('drug', 'N/A')}, "
                f"Condition={row.get('condition', 'N/A')}, "
                f"Phase={row.get('phase', 'N/A')}, "
                f"Status={row.get('status', 'N/A')}"
            )
        return "\n".join(lines)

    return "Please specify a drug, disease, or sponsor to build context."
