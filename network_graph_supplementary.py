"""
Network Graph Supplementary
============================
Advanced graph visualisations for the Clinical Trials Knowledge Graph.

All functions return (fig, df | G) for direct use in Streamlit.
Relies on application.py's KGClient.
"""

import io
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Optional

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

from application import KGClient


# ══════════════════════════════════════════════════════════════
#  DARK STYLE DEFAULTS
# ══════════════════════════════════════════════════════════════

_BG  = "#0F1117"
_FG  = "#CCCCDD"
_ACC = "#7B68EE"   # medium slate blue accent

plt.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor":   _BG,
    "axes.edgecolor":   "#333344",
    "axes.labelcolor":  _FG,
    "text.color":       _FG,
    "xtick.color":      _FG,
    "ytick.color":      _FG,
    "axes.grid":        True,
    "grid.color":       "#1E1E2E",
    "grid.linestyle":   "--",
    "axes.titlecolor":  "#EEEEFF",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})


# ══════════════════════════════════════════════════════════════
#  1. DRUG–CONDITION BIPARTITE SUBGRAPH (ego-graph style)
# ══════════════════════════════════════════════════════════════

def drug_condition_subgraph(kg: KGClient, drug: str, radius: int = 2):
    """
    Draws an ego-graph centred on a drug node showing connected conditions,
    studies, and sponsors up to `radius` hops away.

    Returns (fig, G).
    """
    query = """
    MATCH (i:Intervention)<-[:USES_INTERVENTION]-(st:Study)
    WHERE toLower(i.name) CONTAINS toLower($drug)
    MATCH (st)-[:STUDIES]->(c:Condition)
    OPTIONAL MATCH (st)<-[:SPONSORS]-(s:Sponsor)
    RETURN i.name AS drug, c.name AS condition,
           s.name AS sponsor, COUNT(DISTINCT st.nct_id) AS weight
    ORDER BY weight DESC
    LIMIT 80
    """
    rows = kg.query(query, {"drug": drug})
    if not rows:
        return None, nx.Graph()

    G = nx.Graph()
    drug_node = drug.title()
    G.add_node(drug_node, ntype="drug")

    for r in rows:
        cond = r.get("condition")
        spons = r.get("sponsor")
        w = r.get("weight", 1)
        if cond:
            G.add_node(cond, ntype="condition")
            G.add_edge(drug_node, cond, weight=w)
        if spons and spons not in (None, ""):
            G.add_node(spons, ntype="sponsor")
            if cond:
                G.add_edge(cond, spons, weight=1)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Node colours by type
    color_map = {"drug": "#E63946", "condition": "#457B9D", "sponsor": "#2A9D8F"}
    node_colors = [color_map.get(G.nodes[n].get("ntype", "condition"), "#888") for n in G.nodes()]
    node_sizes  = [3000 if G.nodes[n].get("ntype") == "drug" else 800 for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25,
                           width=[1 + 2 * (w / max_w) for w in edge_weights],
                           edge_color="#444466")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color=_FG, font_weight="bold")

    # Legend
    for label, color in color_map.items():
        ax.scatter([], [], c=color, label=label.title(), s=80)
    ax.legend(loc="upper left", framealpha=0.2, labelcolor=_FG)

    ax.set_title(f"Drug–Condition–Sponsor Subgraph: {drug}", pad=15)
    ax.axis("off")
    plt.tight_layout()
    return fig, G


# ══════════════════════════════════════════════════════════════
#  2. SPONSOR COLLABORATION NETWORK
# ══════════════════════════════════════════════════════════════

def sponsor_network(kg: KGClient, min_shared_trials: int = 3, top_n: int = 60):
    """
    Builds a weighted sponsor co-occurrence graph (sponsors who share trials).
    Node size ∝ total trials; edge weight = shared trials.
    Returns (fig, G).
    """
    query = """
    MATCH (s1:Sponsor)-[:SPONSORS]->(st:Study)<-[:SPONSORS]-(s2:Sponsor)
    WHERE id(s1) < id(s2)
    RETURN s1.name AS sponsor_a, s2.name AS sponsor_b,
           COUNT(DISTINCT st.nct_id) AS shared_trials
    ORDER BY shared_trials DESC
    LIMIT $top_n
    """
    rows = kg.query(query, {"top_n": top_n * 5})
    if not rows:
        return None, nx.Graph()

    df = pd.DataFrame(rows)
    df = df[df["shared_trials"] >= min_shared_trials]

    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(r["sponsor_a"], r["sponsor_b"], weight=int(r["shared_trials"]))

    # Keep only the top_n highest-degree nodes
    if len(G.nodes) > top_n:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]
        G = G.subgraph([n for n, _ in top_nodes]).copy()

    pos = nx.spring_layout(G, seed=7, k=1.5, weight="weight")

    deg = dict(G.degree(weight="weight"))
    max_deg = max(deg.values()) if deg else 1
    node_sizes = [300 + 2000 * (deg[n] / max_deg) for n in G.nodes()]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_ew = max(edge_weights) if edge_weights else 1

    cmap_nodes = cm.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=0, vmax=max_deg)
    node_colors = [cmap_nodes(norm(deg[n])) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3,
                           width=[0.5 + 3 * (w / max_ew) for w in edge_weights],
                           edge_color="#555577")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=5.5, font_color=_FG)

    ax.set_title(f"Sponsor Collaboration Network (min {min_shared_trials} shared trials)", pad=15)
    ax.axis("off")
    plt.tight_layout()
    return fig, G


# ══════════════════════════════════════════════════════════════
#  3. CONDITION SIMILARITY NETWORK (shared drugs)
# ══════════════════════════════════════════════════════════════

def condition_similarity_network(kg: KGClient, top_n: int = 40,
                                  min_shared: int = 2):
    """
    Builds a condition–condition graph where edge weight = number of shared drugs.
    Reveals which diseases are clinically adjacent (treated by same agents).
    Returns (fig, G).
    """
    query = """
    MATCH (c1:Condition)<-[:STUDIES]-(st1:Study)-[:USES_INTERVENTION]->(i:Intervention)
         <-[:USES_INTERVENTION]-(st2:Study)-[:STUDIES]->(c2:Condition)
    WHERE id(c1) < id(c2)
    RETURN c1.name AS cond_a, c2.name AS cond_b,
           COUNT(DISTINCT i.name) AS shared_drugs
    ORDER BY shared_drugs DESC
    LIMIT 500
    """
    rows = kg.query(query)
    if not rows:
        return None, nx.Graph()

    df = pd.DataFrame(rows)
    df = df[df["shared_drugs"] >= min_shared]

    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(r["cond_a"], r["cond_b"], weight=int(r["shared_drugs"]))

    if len(G.nodes) > top_n:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]
        G = G.subgraph([n for n, _ in top_nodes]).copy()

    pos = nx.spring_layout(G, seed=3, k=2.0, weight="weight")

    deg = dict(G.degree())
    max_deg = max(deg.values()) if deg else 1
    node_sizes = [200 + 1500 * (deg[n] / max_deg) for n in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_ew = max(edge_weights) if edge_weights else 1

    cmap = cm.get_cmap("cool")
    norm = mcolors.Normalize(vmin=0, vmax=max_deg)
    node_colors = [cmap(norm(deg[n])) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25,
                           width=[0.3 + 2 * (w / max_ew) for w in edge_weights],
                           edge_color="#334455")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=5.5, font_color=_FG)

    ax.set_title(f"Condition Similarity Network (min {min_shared} shared drugs)", pad=15)
    ax.axis("off")
    plt.tight_layout()
    return fig, G


# ══════════════════════════════════════════════════════════════
#  4. GRAPH METRICS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════

def graph_metrics(kg: KGClient) -> pd.DataFrame:
    """
    Computes basic graph metrics for the drug–condition bipartite graph.
    Returns a summary DataFrame.
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

    components = list(nx.connected_components(G))
    largest = max(components, key=len)

    metrics = {
        "Nodes": [G.number_of_nodes()],
        "Edges": [G.number_of_edges()],
        "Components": [len(components)],
        "Largest Component Size": [len(largest)],
        "Avg Degree": [round(sum(d for _, d in G.degree()) / G.number_of_nodes(), 2)],
        "Density": [round(nx.density(G), 6)],
    }
    return pd.DataFrame(metrics).T.rename(columns={0: "Value"})


# ══════════════════════════════════════════════════════════════
#  5. DEGREE DISTRIBUTION
# ══════════════════════════════════════════════════════════════

def degree_distribution(kg: KGClient):
    """
    Plots the degree distribution of the drug–condition bipartite graph.
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

    degrees = sorted([d for _, d in G.degree()], reverse=True)
    df = pd.DataFrame({"degree": degrees})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(_BG)

    axes[0].hist(degrees, bins=50, color=_ACC, edgecolor=_BG, log=True)
    axes[0].set_title("Degree Distribution (log scale)")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count (log)")

    deg_count = pd.Series(degrees).value_counts().sort_index()
    axes[1].loglog(deg_count.index, deg_count.values, "o", color="#E63946",
                   markersize=4, alpha=0.7)
    axes[1].set_title("Degree Distribution (log–log)")
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Count (log)")

    plt.suptitle("Drug–Condition Graph Degree Distribution",
                 color="#EEEEFF", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, df


# ══════════════════════════════════════════════════════════════
#  6. BRIDGE DRUGS (structural holes)
# ══════════════════════════════════════════════════════════════

def bridge_drugs(kg: KGClient, top_n: int = 20):
    """
    Identifies drugs that act as bridges between disease communities
    using betweenness centrality on the full graph.
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

    bet = nx.betweenness_centrality(G, k=min(300, len(G)))
    # Keep only drug nodes
    drug_nodes = {n for n, d in G.nodes(data=True)
                  if not any(c.isdigit() for c in n[:3])}
    # Approximate: keep nodes that are connected to condition-like counterparts
    bet_df = pd.DataFrame([
        {"node": n, "betweenness": v}
        for n, v in bet.items()
    ]).sort_values("betweenness", ascending=False).head(top_n * 2)
    bet_df = bet_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(_BG)
    bet_df.plot.barh(x="node", y="betweenness", legend=False,
                     color="#E63946", edgecolor="none", ax=ax)
    ax.set_title(f"Top {top_n} Bridge Nodes (Betweenness Centrality)")
    ax.set_xlabel("Betweenness Centrality")
    plt.tight_layout()
    return fig, bet_df
