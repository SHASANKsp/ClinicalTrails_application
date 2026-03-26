"""
Clinical Trials Knowledge Graph — Analytics Applications
=========================================================
20 applications across 5 categories:
  1. Drug Intelligence      (4 apps)
  2. Disease Analytics      (5 apps)
  3. Sponsor Intelligence   (5 apps)
  4. Network & Graph        (3 apps)
  5. Geo & Temporal         (3 apps)

Each function:
  - Runs a Cypher query against the KG
  - Saves a CSV to  outputs/<app>/data/
  - Saves a PNG to  outputs/<app>/plots/   (where applicable)
  - Returns the raw DataFrame for further use / LLM piping
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import networkx as nx
import community as community_louvain   # pip install python-louvain
from collections import Counter
from neo4j import GraphDatabase

BASE_DIR = "outputs"
os.makedirs(BASE_DIR, exist_ok=True)

URI = "neo4j://localhost:7687" 
USER = "neo4j"
PASSWORD = "CTrail@123"

# Step 2: Initialize KG client
kg = KGClient(URI, USER, PASSWORD)

# ─────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────

def _dirs(app_name):
    base     = os.path.join(BASE_DIR, app_name)
    data_dir = os.path.join(base, "data")
    plot_dir = os.path.join(base, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return data_dir, plot_dir

def _save(df, data_dir, filename):
    path = os.path.join(data_dir, filename)
    df.to_csv(path, index=False)
    print(f"  [csv]  {path}  ({len(df)} rows)")

def _savefig(plot_dir, filename):
    path = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot] {path}")


# ══════════════════════════════════════════════════════════════
#  CATEGORY 1 — DRUG INTELLIGENCE
# ══════════════════════════════════════════════════════════════

def drug_evidence(kg, drug):
    """
    Full evidence profile for a drug:
    trials, phases, conditions, sponsors, countries.
    Validates: Are phases and conditions populated?
    """
    data_dir, plot_dir = _dirs("drug_evidence")

    query = """
    MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)
    WHERE toLower(i.name) CONTAINS toLower($drug)
    OPTIONAL MATCH (st)-[:STUDIES]->(c:Condition)
    OPTIONAL MATCH (st)<-[:SPONSORS]-(s:Sponsor)
    OPTIONAL MATCH (st)-[:CONDUCTED_AT]->(l:Location)
    RETURN st.nct_id       AS trial,
           st.phases        AS phase,
           st.overall_status AS status,
           st.enrollment     AS enrollment,
           c.name            AS condition,
           s.name            AS sponsor,
           l.country         AS country
    """
    df = pd.DataFrame(kg.query(query, {"drug": drug}))
    _save(df, data_dir, f"{drug}.csv")

    # --- plot 1: phase distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    phase_counts = df["phase"].value_counts()
    phase_counts.plot.bar(ax=axes[0], color="steelblue")
    axes[0].set_title(f"{drug} – Phase distribution")
    axes[0].set_xlabel("Phase")
    axes[0].set_ylabel("Trials")
    axes[0].tick_params(axis="x", rotation=45)

    status_counts = df["status"].value_counts().head(8)
    status_counts.plot.barh(ax=axes[1], color="darkcyan")
    axes[1].set_title(f"{drug} – Trial status")
    axes[1].set_xlabel("Count")

    _savefig(plot_dir, f"{drug}_evidence.png")
    return df


def drug_competition(kg, drug):
    """
    Competing drugs in the same conditions as the target drug.
    Useful for market analysis and pipeline differentiation.
    """
    data_dir, plot_dir = _dirs("drug_competition")

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
    _save(df, data_dir, f"{drug}.csv")

    df.head(15).plot.bar(x="competitor", y="trials", legend=False, color="tomato")
    plt.title(f"Top competitors to {drug} (shared conditions)")
    plt.xlabel("Drug")
    plt.ylabel("Trials")
    plt.xticks(rotation=45, ha="right")
    _savefig(plot_dir, f"{drug}_competition.png")
    return df


def drug_geo(kg, drug):
    """
    Countries conducting trials for a drug.
    Validates: geographic spread, identifying dominant markets.
    """
    data_dir, plot_dir = _dirs("drug_geo")

    query = """
    MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)-[:CONDUCTED_AT]->(l:Location)
    WHERE toLower(i.name) CONTAINS toLower($drug)
    RETURN l.country AS country, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    """
    df = pd.DataFrame(kg.query(query, {"drug": drug}))
    _save(df, data_dir, f"{drug}.csv")

    df.head(20).plot.bar(x="country", y="trials", legend=False, color="mediumseagreen")
    plt.title(f"{drug} – Trial countries")
    plt.xlabel("Country")
    plt.ylabel("Trials")
    plt.xticks(rotation=45, ha="right")
    _savefig(plot_dir, f"{drug}_geo.png")
    return df


def drug_paths(kg, drug):
    """
    Multi-hop paths: Drug → Study → Condition + Sponsor.
    Useful for building subgraphs fed into a GraphRAG pipeline.
    """
    data_dir, _ = _dirs("drug_paths")

    query = """
    MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)
    WHERE toLower(i.name) CONTAINS toLower($drug)
    MATCH (st)-[:STUDIES]->(c:Condition)
    MATCH (st)<-[:SPONSORS]-(s:Sponsor)
    RETURN i.name   AS drug,
           st.nct_id AS trial,
           c.name    AS condition,
           s.name    AS sponsor,
           s.class   AS sponsor_class
    """
    df = pd.DataFrame(kg.query(query, {"drug": drug}))
    _save(df, data_dir, f"{drug}.csv")
    return df


# ══════════════════════════════════════════════════════════════
#  CATEGORY 2 — DISEASE ANALYTICS
# ══════════════════════════════════════════════════════════════

def disease_landscape(kg, disease):
    """
    All drugs being trialled for a disease, ranked by trial count.
    Reveals therapeutic crowding vs opportunity gaps.
    """
    data_dir, plot_dir = _dirs("disease_landscape")

    query = """
    MATCH (st:Study)-[:STUDIES]->(c:Condition)
    WHERE toLower(c.name) CONTAINS toLower($disease)
    MATCH (st)-[:USES_INTERVENTION]->(i:Intervention)
    RETURN i.name AS drug, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    LIMIT 30
    """
    df = pd.DataFrame(kg.query(query, {"disease": disease}))
    _save(df, data_dir, f"{disease}.csv")

    df.head(15).plot.barh(x="drug", y="trials", legend=False, color="slateblue")
    plt.title(f"{disease} – Treatment landscape (top drugs)")
    plt.xlabel("Trials")
    _savefig(plot_dir, f"{disease}_landscape.png")
    return df


def disease_design(kg, disease):
    """
    Trial design breakdown: arm types, allocation, masking.
    Validates: what study designs dominate in this disease area?
    """
    data_dir, plot_dir = _dirs("disease_design")

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
    _save(df, data_dir, f"{disease}.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["allocation", "masking", "arm_type"]):
        counts = df.groupby(col)["count"].sum().sort_values(ascending=False).head(8)
        counts.plot.bar(ax=ax, color="mediumpurple")
        ax.set_title(col.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(f"{disease} – Trial design patterns")
    _savefig(plot_dir, f"{disease}_design.png")
    return df


def disease_phase_progression(kg, disease):
    """
    Phase distribution across all trials for a disease.
    Shows pipeline maturity (Phase I vs III balance).
    """
    data_dir, plot_dir = _dirs("disease_phase")

    query = """
    MATCH (st:Study)-[:STUDIES]->(c:Condition)
    WHERE toLower(c.name) CONTAINS toLower($disease)
    RETURN st.phases AS phase, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY phase
    """
    df = pd.DataFrame(kg.query(query, {"disease": disease}))
    _save(df, data_dir, f"{disease}.csv")

    df.plot.bar(x="phase", y="trials", legend=False, color="mediumpurple")
    plt.title(f"{disease} – Phase distribution")
    plt.xlabel("Phase")
    plt.ylabel("Trials")
    plt.xticks(rotation=45, ha="right")
    _savefig(plot_dir, f"{disease}_phase.png")
    return df


def disease_enrollment(kg, disease):
    """
    Enrollment size distribution for a disease.
    Identifies large-scale vs early/small trials.
    """
    data_dir, plot_dir = _dirs("disease_enrollment")

    query = """
    MATCH (st:Study)-[:STUDIES]->(c:Condition)
    WHERE toLower(c.name) CONTAINS toLower($disease)
      AND st.enrollment IS NOT NULL
    RETURN st.nct_id    AS trial,
           st.enrollment AS enrollment,
           st.phases     AS phase
    """
    df = pd.DataFrame(kg.query(query, {"disease": disease}))
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df = df.dropna(subset=["enrollment"])
    _save(df, data_dir, f"{disease}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["enrollment"].clip(upper=df["enrollment"].quantile(0.95)).hist(
        bins=30, ax=axes[0], color="steelblue", edgecolor="white"
    )
    axes[0].set_title(f"{disease} – Enrollment size (≤95th pct)")
    axes[0].set_xlabel("Participants")

    phase_enroll = df.groupby("phase")["enrollment"].median().sort_values()
    phase_enroll.plot.barh(ax=axes[1], color="steelblue")
    axes[1].set_title("Median enrollment by phase")
    axes[1].set_xlabel("Median enrollment")

    plt.suptitle(f"{disease} – Enrollment analysis")
    _savefig(plot_dir, f"{disease}_enrollment.png")
    return df


def disease_sponsor_diversity(kg, disease):
    """
    Who is sponsoring trials for a disease and what class are they?
    Industry vs Academic vs Government split.
    """
    data_dir, plot_dir = _dirs("disease_sponsor_diversity")

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
    _save(df, data_dir, f"{disease}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Top sponsors
    df.head(12).plot.barh(x="sponsor", y="trials", ax=axes[0],
                          legend=False, color="darkcyan")
    axes[0].set_title(f"Top sponsors for {disease}")

    # Sponsor class breakdown
    class_counts = df.groupby("sponsor_class")["trials"].sum().sort_values()
    class_counts.plot.barh(ax=axes[1], color="teal")
    axes[1].set_title("By sponsor class")

    plt.suptitle(f"{disease} – Sponsor diversity")
    _savefig(plot_dir, f"{disease}_sponsor_diversity.png")
    return df


# ══════════════════════════════════════════════════════════════
#  CATEGORY 3 — SPONSOR INTELLIGENCE
# ══════════════════════════════════════════════════════════════

def sponsor_portfolio(kg, sponsor):
    """
    Condition portfolio of a sponsor: where are they investing?
    """
    data_dir, plot_dir = _dirs("sponsor_portfolio")

    query = """
    MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
    WHERE toLower(s.name) CONTAINS toLower($sponsor)
    MATCH (st)-[:STUDIES]->(c:Condition)
    RETURN c.name AS condition, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    LIMIT 30
    """
    df = pd.DataFrame(kg.query(query, {"sponsor": sponsor}))
    _save(df, data_dir, f"{sponsor}.csv")

    df.head(15).plot.barh(x="condition", y="trials", legend=False, color="goldenrod")
    plt.title(f"{sponsor} – Condition portfolio")
    plt.xlabel("Trials")
    _savefig(plot_dir, f"{sponsor}_portfolio.png")
    return df


def sponsor_geo(kg, sponsor):
    """
    Geographic distribution of a sponsor's trials.
    """
    data_dir, plot_dir = _dirs("sponsor_geo")

    query = """
    MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
    WHERE toLower(s.name) CONTAINS toLower($sponsor)
    MATCH (st)-[:CONDUCTED_AT]->(l:Location)
    RETURN l.country AS country, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    """
    df = pd.DataFrame(kg.query(query, {"sponsor": sponsor}))
    _save(df, data_dir, f"{sponsor}.csv")

    df.head(20).plot.bar(x="country", y="trials", legend=False, color="orange")
    plt.title(f"{sponsor} – Geographic reach")
    plt.xticks(rotation=45, ha="right")
    _savefig(plot_dir, f"{sponsor}_geo.png")
    return df


def sponsor_pipeline(kg, sponsor):
    """
    Phase breakdown of a sponsor's pipeline.
    Shows early-stage vs late-stage investment mix.
    """
    data_dir, plot_dir = _dirs("sponsor_pipeline")

    query = """
    MATCH (st:Study)<-[:SPONSORS]-(s:Sponsor)
    WHERE toLower(s.name) CONTAINS toLower($sponsor)
    RETURN st.phases        AS phase,
           st.overall_status AS status,
           COUNT(DISTINCT st.nct_id) AS trials
    """
    df = pd.DataFrame(kg.query(query, {"sponsor": sponsor}))
    _save(df, data_dir, f"{sponsor}.csv")

    pivot = df.groupby(["phase", "status"])["trials"].sum().unstack(fill_value=0)
    pivot.plot.bar(stacked=True, figsize=(12, 5), colormap="tab10")
    plt.title(f"{sponsor} – Pipeline by phase & status")
    plt.xlabel("Phase")
    plt.ylabel("Trials")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="upper right", fontsize=8)
    _savefig(plot_dir, f"{sponsor}_pipeline.png")
    return df


def sponsor_drugs(kg, sponsor):
    """
    All interventions a sponsor is running trials for.
    """
    data_dir, plot_dir = _dirs("sponsor_drugs")

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
    _save(df, data_dir, f"{sponsor}.csv")

    df.head(15).plot.barh(x="drug", y="trials", legend=False, color="darkorange")
    plt.title(f"{sponsor} – Drug pipeline")
    plt.xlabel("Trials")
    _savefig(plot_dir, f"{sponsor}_drugs.png")
    return df


def sponsor_collaborators(kg, sponsor):
    """
    Other sponsors who appear in the same trials as the target sponsor.
    Reveals co-development partnerships and research consortia.
    """
    data_dir, plot_dir = _dirs("sponsor_collaborators")

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
    _save(df, data_dir, f"{sponsor}.csv")

    if not df.empty:
        df.head(15).plot.barh(x="collaborator", y="shared_trials",
                               legend=False, color="peru")
        plt.title(f"{sponsor} – Frequent collaborators")
        plt.xlabel("Shared trials")
        _savefig(plot_dir, f"{sponsor}_collaborators.png")
    return df


# ══════════════════════════════════════════════════════════════
#  CATEGORY 4 — NETWORK & GRAPH ANALYTICS
# ══════════════════════════════════════════════════════════════

def centrality(kg):
    """
    Degree centrality on the drug–condition bipartite graph.
    High-centrality drugs = broad-spectrum agents.
    High-centrality conditions = heavily studied diseases.
    """
    data_dir, plot_dir = _dirs("network_centrality")

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

    deg  = nx.degree_centrality(G)
    bet  = nx.betweenness_centrality(G, k=min(200, len(G)))
    df = pd.DataFrame({
        "node": list(deg.keys()),
        "degree_centrality":      [deg[n] for n in deg],
        "betweenness_centrality": [bet.get(n, 0) for n in deg],
    }).sort_values("degree_centrality", ascending=False)

    _save(df, data_dir, "centrality.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df.head(20).plot.barh(x="node", y="degree_centrality",
                           ax=axes[0], legend=False, color="purple")
    axes[0].set_title("Top 20 by degree centrality")

    df.sort_values("betweenness_centrality", ascending=False).head(20).plot.barh(
        x="node", y="betweenness_centrality",
        ax=axes[1], legend=False, color="indigo"
    )
    axes[1].set_title("Top 20 by betweenness centrality")

    plt.suptitle("Drug–Condition network centrality")
    _savefig(plot_dir, "centrality.png")
    return df


def community_detection(kg):
    """
    Louvain community detection on the drug–condition graph.
    Clusters reveal disease areas + their dominant drugs.
    Requires: pip install python-louvain networkx
    """
    data_dir, plot_dir = _dirs("network_communities")

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

    # Summarise each community
    community_sizes = df["community"].value_counts().reset_index()
    community_sizes.columns = ["community", "members"]
    _save(df, data_dir, "communities_raw.csv")
    _save(community_sizes, data_dir, "community_sizes.csv")

    community_sizes.head(20).plot.bar(x="community", y="members",
                                       legend=False, color="slateblue")
    plt.title("Community sizes (Louvain on drug–condition graph)")
    plt.xlabel("Community ID")
    plt.ylabel("Nodes")
    _savefig(plot_dir, "communities.png")
    return df


def drug_repurposing(kg):
    """
    Drugs studied across the largest number of distinct conditions.
    High overlap = potential repurposing candidates for GraphRAG.
    """
    data_dir, plot_dir = _dirs("drug_repurposing")

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
    _save(df, data_dir, "repurposing_candidates.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(df["n_trials"], df["n_conditions"], alpha=0.7, color="darkorchid", s=60)
    for _, row in df.head(15).iterrows():
        ax.annotate(row["drug"], (row["n_trials"], row["n_conditions"]),
                    fontsize=7, alpha=0.8)
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Number of distinct conditions")
    ax.set_title("Drug repurposing signals (breadth of conditions studied)")
    _savefig(plot_dir, "repurposing.png")
    return df


# ══════════════════════════════════════════════════════════════
#  CATEGORY 5 — GEO & TEMPORAL ANALYTICS
# ══════════════════════════════════════════════════════════════

def trial_density(kg):
    """
    Global trial density by country.
    """
    data_dir, plot_dir = _dirs("geo_density")

    query = """
    MATCH (st:Study)-[:CONDUCTED_AT]->(l:Location)
    RETURN l.country AS country, COUNT(DISTINCT st.nct_id) AS trials
    ORDER BY trials DESC
    """
    df = pd.DataFrame(kg.query(query))
    _save(df, data_dir, "global.csv")

    df.head(25).plot.bar(x="country", y="trials", legend=False, color="teal")
    plt.title("Global trial density by country")
    plt.xticks(rotation=45, ha="right")
    _savefig(plot_dir, "density.png")
    return df


def trial_timeline(kg):
    """
    Trial start date distribution over time.
    Shows activity waves, COVID dips, research surges.
    """
    data_dir, plot_dir = _dirs("geo_timeline")

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
    _save(df, data_dir, "timeline.csv")

    # Aggregate by year-month
    df["ym"] = df["start_date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("ym")["trials"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly["ym"], monthly["trials"], linewidth=1.5, color="steelblue")
    ax.fill_between(monthly["ym"], monthly["trials"], alpha=0.2, color="steelblue")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Trial starts over time (monthly)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trials started")
    plt.xticks(rotation=45)
    _savefig(plot_dir, "timeline.png")
    return df


def geo_phase_heatmap(kg):
    """
    Pivot: country × phase trial counts.
    Reveals which countries run late-phase (Phase III) trials.
    """
    data_dir, plot_dir = _dirs("geo_phase_heatmap")

    query = """
    MATCH (st:Study)-[:CONDUCTED_AT]->(l:Location)
    WHERE l.country IS NOT NULL AND st.phases IS NOT NULL
    RETURN l.country AS country, st.phases AS phase,
           COUNT(DISTINCT st.nct_id) AS trials
    """
    df = pd.DataFrame(kg.query(query))
    _save(df, data_dir, "geo_phase.csv")

    pivot = (
        df.groupby(["country", "phase"])["trials"]
          .sum()
          .unstack(fill_value=0)
    )
    top_countries = pivot.sum(axis=1).nlargest(20).index
    pivot = pivot.loc[top_countries]

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.3, cbar_kws={"label": "Trials"})
    plt.title("Country × Phase heatmap (top 20 countries)")
    plt.tight_layout()
    _savefig(plot_dir, "geo_phase_heatmap.png")
    return df


# ══════════════════════════════════════════════════════════════
#  RUNNER — execute all applications
# ══════════════════════════════════════════════════════════════

def run_all(kg,
            drug="Donepezil",
            disease="Alzheimer",
            sponsor="Pfizer"):
    """
    Run every application in sequence.
    Outputs land in outputs/<app>/data/ and outputs/<app>/plots/.
    """

    print("\n══ 1. DRUG INTELLIGENCE ══════════════════")
    drug_evidence(kg, drug)
    drug_competition(kg, drug)
    drug_geo(kg, drug)
    drug_paths(kg, drug)

    print("\n══ 2. DISEASE ANALYTICS ══════════════════")
    disease_landscape(kg, disease)
    disease_design(kg, disease)
    disease_phase_progression(kg, disease)
    disease_enrollment(kg, disease)
    disease_sponsor_diversity(kg, disease)

    print("\n══ 3. SPONSOR INTELLIGENCE ═══════════════")
    sponsor_portfolio(kg, sponsor)
    sponsor_geo(kg, sponsor)
    sponsor_pipeline(kg, sponsor)
    sponsor_drugs(kg, sponsor)
    sponsor_collaborators(kg, sponsor)

    print("\n══ 4. NETWORK & GRAPH ANALYTICS ══════════")
    centrality(kg)
    community_detection(kg)
    drug_repurposing(kg)

    print("\n══ 5. GEO & TEMPORAL ANALYTICS ═══════════")
    trial_density(kg)
    trial_timeline(kg)
    geo_phase_heatmap(kg)

    print("\n✅  All 20 applications executed.")
    print(f"    CSVs  → outputs/<app>/data/")
    print(f"    Plots → outputs/<app>/plots/")


# ── entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    # Replace `kg` with your Neo4j graph connection object.
    # e.g.  from your_graph_module import KnowledgeGraph
    #        kg = KnowledgeGraph(uri=..., user=..., password=...)
    run_all(kg)
