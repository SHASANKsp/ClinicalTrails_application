# ClinicalTrials Knowledge Graph

A graph native intelligence platform built on Neo4j that transforms raw ClinicalTrials.gov data into a queryable knowledge graph, powering drug intelligence, disease analytics, sponsor profiling, network analysis, and a GraphRAG pipeline for grounded clinical Q&A.

## Overview

Clinical trial registries contain huge data points on drugs, diseases, sponsors, locations, and outcomes, but they are hard to query across dimensions. This project constructs a **property graph** where studies, interventions, conditions, sponsors, arms, locations, and outcomes are nodes connected by semantically meaningful relationships.

The result is a platform that can answers questions like:

- *"What drugs compete with Semaglutide, broken down by trial volume and geography?"*
- *"How does the Alzheimer's pipeline funnel compare to Diabetes?"*
- *"Which drugs appear as structural bridges across disease communities?"*
- *"What are Pfizer's top therapeutic areas and phase completion rates?"*

---

## Architecture

```
ClinicalTrials.gov Data
        │
        ▼
  CSV Extraction & Cleaning
        │
        ▼
  Neo4j Knowledge Graph  ◄──── injection.py
        │
        ▼
  Defined functions for each application
  ├── Drug Intelligence Queries
  ├── Disease Analytics Queries
  ├── Sponsor Intelligence Queries
  └── Network & Graph Analytics
        │
        ▼
  GraphRAG Pipeline
  └── LLM · Summarization · Q&A
```

---

## Graph Schema

### Node Labels

| Label  | Description |
|---|---|
| `Study` | A registered clinical trial |
| `Intervention` | Drug, device, or procedure tested |
| `Condition` | Disease or condition studied |
| `Sponsor` | Organization funding/running the trial |
| `Arm` | Trial arm (experimental, placebo, comparator) |
| `Location` | Facility conducting the trial |

### Relationships

| Relationship | Direction | Meaning |
|---|---|---|
| `SPONSORS` | Sponsor → Study | Sponsor funds the study |
| `STUDIES` | Study → Condition | Study targets a disease |
| `USES_INTERVENTION` | Study → Intervention | Study tests a drug/device |
| `HAS_ARM` | Study → Arm | Study contains a trial arm |
| `CONDUCTED_AT` | Study → Location | Study runs at a facility |

---

## Modules

### 01 · Drug Intelligence

Query any intervention's full evidence profile in a single graph traversal.

| Feature | What It Does |
|---|---|
| **Evidence Profile** | Aggregates all trials by phase, status, condition, and geography | 
| **Competitive Landscape** | Finds all drugs sharing a condition with the target compound, ranked by trial volume | 
| **Geographic Footprint** | Maps countries conducting trials for a drug | 
| **Multi-hop Paths** | Constructs Drug → Study → Condition → Sponsor subgraphs for GraphRAG context |

**Example: Semaglutide -** Phase 3 heavy, US centric geography, Novo Nordisk as lead sponsor, Diabetes Mellitus Type 2 and Obesity as primary conditions.

**Example: Donepezil -** Competitor list spans oncology and antipsychotics (reflecting Alzheimer's comorbidity landscape), Phase 2 peak signals an unmet need space still searching for breakthrough efficacy.

---

### 02 · Disease Analytics

Characterize the full clinical landscape for any disease area.

| Feature | What It Does |
|---|---|
| **Treatment Landscape** | Ranks all active drugs by trial count |
| **Trial Design Patterns** | Breaks down allocation, blinding, and arm types | 
| **Phase Progression** | Plots the trial funnel by phase; inversion signals high attrition | 
| **Enrollment Analysis** | Histogram + phase stratified median charts | 
| **Sponsor Diversity** | Classifies sponsors as Industry / Academic / Government | 

**Example: Diabetes -** Mature, RCT dominated, >85% randomized, Phase 3 enrollment medians exceeding 300 participants. Industry led by Novo Nordisk and Eli Lilly.

**Example: Alzheimer's -** Inverted funnel peaking at Phase 2, high academic/government sponsorship, large Phase 3 enrollment requirements signaling high cost and risk.

---

### 03 · Sponsor Intelligence

Profile any sponsor's therapeutic footprint, pipeline health, and partnership network.

| Feature | What It Does |
|---|---|
| **Condition Portfolio** | Ranks every disease area a sponsor runs trials in |
| **Geographic Reach** | Maps countries where the sponsor conducts trials |
| **Pipeline Mix** | Cross tabulates phase × status in a stacked bar |
| **Drug Inventory** | Lists every intervention a sponsor trials, ranked by volume |
| **Collaborator Network** | Identifies co-sponsors from shared trials |

**Example: Pfizer -** 15+ therapeutic areas, Phase 2/3 completion heavy pipeline, US and UK leading geographic reach.

**Example: Novo Nordisk -** Tightly focused on Diabetes and Obesity, Semaglutide and insulin analogues as flagship compounds, Germany topping geographic reach.

---

### 04 · Network & Graph Analytics

Applies graph algorithms to reveal structural patterns invisible to tabular analysis.

| Feature | What It Does | Algorithm |
|---|---|---|
| **Centrality Scoring** | Computes degree and betweenness centrality across the drug condition graph | Degree / Betweenness |
| **Community Detection** | Partitions the graph into dense therapeutic clusters | Louvain |
| **Drug Repurposing Signals** | Identifies drugs studied across the greatest number of distinct conditions relative to trial volume | Condition breadth scoring |

**Centrality highlights:**
- **Obesity** - highest degree node; GLP-1, SGLT2, bariatric, and behavioral trials all converge here
- **Dexamethasone** - sole pharmaceutical in the betweenness top 25; bridges oncology, COVID19, surgery, sepsis, and haematology simultaneously

**Community detection:** 15,585 total communities detected; 28 mega clusters (>1,000 nodes) align precisely with real biomedical domains including Metabolic & Behavioral Health, Oncology, Cardiovascular, Infectious Disease, Hepatitis B Virology, with no expert labeling required.

**Repurposing archetypes identified:**
- Cytotoxic backbone drugs (broad oncology use via DNA damage + immune modulation)
- Checkpoint inhibitors (tumor agnostic expansion; Nivolumab appearing in HBV and COVID-19)
- Corticosteroids (Dexamethasone as the canonical cross disease repurposing success)

---

### 05 · GraphRAG Pipeline

The KG is designed as the retrieval backbone for a grounded clinical Q&A system.

```
User Query
    │
    ▼
Named Entity Recognition (drug / disease / sponsor)
    │
    ▼
Neo4j Subgraph Extraction
(Multi-hop paths: Drug → Study → Condition → Sponsor)
    │
    ▼
Community Summary Injection
(Louvain cluster context chunks)
    │
    ▼
LLM (Claude / GPT)
    │
    ▼
Grounded Answer with Citation Ready Evidence Chains
```

This architecture grounds every LLM response in structured, citation traceable graph paths, eliminating hallucination of trial statistics and sponsor relationships.

---

## Key Findings

| Domain | Finding |
|---|---|
| Drug | Semaglutide is the most Phase 3 validated GLP-1 agonist; US centric geography is a generalizability risk |
| Drug | Donepezil's trial landscape spans oncology and antipsychotics, reflecting the Alzheimer's comorbidity footprint |
| Disease | Diabetes is a benchmark RCT dominated disease; Alzheimer's shows an inverted funnel still in discovery phase |
| Sponsor | Pfizer has the broadest portfolio; Novo Nordisk is the most focused metabolic disease sponsor |
| Graph | Dexamethasone is the only pharmaceutical acting as a cross domain structural bridge in the trial network |
| Graph | Louvain algorithm independently recovers clinically coherent therapeutic communities without expert labels |
| Repurposing | Nivolumab's appearance in HBV and COVID19 trials surfaces as a biological signal for cross disease immune axis activity |

---

## Agentic AI Integration

This KG is built to serve as the structured memory layer for clinical AI agents. Intended integration points:

- **GraphRAG Q&A** — Subgraph extraction feeds structured context to an LLM for grounded natural language answers
- **Hypothesis Generation** — Community detection outputs seed prompts for drug repurposing hypotheses
- **Automated Evidence Summaries** — Multi-hop paths generate citation ready evidence chains for regulatory or medical affairs use cases
- **Tool augmented Agents** — Neo4j can be exposed as an MCP (Model Context Protocol) server, allowing agents to query the KG dynamically via Cypher tool calls
