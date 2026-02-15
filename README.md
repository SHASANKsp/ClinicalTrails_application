# ClinicalTrials Knowledge Graph

## Overview

This project builds a structured, ontology-aware knowledge graph from ClinicalTrials.gov data. It extracts normalized entities from raw clinical trial records and constructs a Neo4j graph designed to support advanced scientific, regulatory, and pharmaceutical analytics.

The system is built for scale and preserves relational structure, temporal attributes, and ontology hierarchy. It enables modeling of trial risk, drug repurposing, competitive landscapes, safety profiling, and innovation dynamics across therapeutic areas.

---

## Objectives

* Transform ClinicalTrials.gov JSON data into normalized, graph-ready entities.
* Preserve structural relationships between studies, arms, interventions, conditions, outcomes, sponsors, and locations.
* Enable ontology-aware reasoning using MeSH hierarchies.
* Support analytical and predictive applications across scientific and industry use cases.
* Provide a scalable ingestion pipeline compatible with large datasets.

---

## System Architecture

The project consists of three main layers:

### 1. Extraction Layer

A streaming JSON parser extracts normalized entity tables from raw ClinicalTrials.gov dumps. Extraction is memory-safe and avoids full in-memory loading of large datasets.

Generated tables include:

* `studies.csv`
* `arms.csv`
* `interventions.csv`
* `conditions.csv`
* `sponsors.csv`
* `locations.csv`
* `outcomes.csv`
* `adverse_events.csv`
* `participant_flow.csv`
* `mesh_ancestors.csv` (if extracted)

Each table represents a distinct entity or relationship component, designed for direct graph ingestion.

---

### 2. Knowledge Graph Schema

The Neo4j graph is structured around the following node types:

**Core Nodes**

* `Study`
* `Arm`
* `Intervention`
* `Condition`
* `Sponsor`
* `Outcome`
* `OutcomeResult`
* `AdverseEvent`
* `Location`
* `MeSH`

**Primary Relationships**

* `(Sponsor)-[:SPONSORS]->(Study)`
* `(Study)-[:HAS_ARM]->(Arm)`
* `(Arm)-[:USES_INTERVENTION]->(Intervention)`
* `(Study)-[:STUDIES]->(Condition)`
* `(Condition)-[:HAS_MESH]->(MeSH)`
* `(MeSH)-[:IS_A]->(MeSH)` (hierarchical ontology)
* `(Study)-[:MEASURES]->(Outcome)`
* `(Outcome)-[:HAS_RESULT]->(OutcomeResult)`
* `(Study)-[:REPORTS_EVENT]->(AdverseEvent)`
* `(Study)-[:CONDUCTED_AT]->(Location)`

This schema preserves:

* Structural relationships
* Trial design information
* Ontological hierarchy
* Temporal attributes
* Quantitative result data

---

### 3. Graph Construction Layer

A Python-based Neo4j ingestion script:

* Creates constraints to ensure uniqueness
* Loads node entities
* Establishes relationships
* Supports batch ingestion
* Avoids duplication via MERGE operations

For large datasets, the design supports migration to Neo4j bulk import methods.

---

## Supported Applications

The graph supports a wide range of scientific and industry applications, including:

### Scientific and Translational Research

* Endpoint evolution analysis
* Cross-phase effect stability assessment
* Rare disease research gap detection
* Eligibility bias modeling
* Mechanism-of-action clustering
* Therapeutic distance modeling using MeSH hierarchy

### Pharmaceutical Strategy

* Failure risk heatmaps by phase and sponsor type
* Drug repurposing detection across disease families
* Competitive landscape analysis
* Sponsor diversification metrics
* Innovation velocity tracking
* Market saturation scoring

### Regulatory and Policy Analytics

* Results reporting delay analysis
* Enrollment inflation trends
* Geographic migration of trials
* Transparency compliance scoring

---

## Technical Stack

* Python
* Pandas (controlled preprocessing)
* ijson (streaming JSON parsing)
* Neo4j
* Neo4j Python Driver
* Cypher query language

Designed for compatibility with large-scale datasets and extensible to graph analytics frameworks.

---

## Installation

### Requirements

* Python 3.9+
* Neo4j 5.x or Aura instance
* Required Python packages:

```bash
pip install neo4j pandas ijson tqdm
```

---

## Usage

### Step 1: Extract Normalized Tables

```bash
python streaming_extractor.py
```

This generates all entity tables under `kg_output/`.

### Step 2: Build Neo4j Knowledge Graph

```bash
python neo4j_loader.py
```

This script:

* Creates node constraints
* Loads entities
* Establishes relationships

---

## Scalability Considerations

For large datasets:

* Use streaming extraction only.
* Avoid full dataframe merges.
* Aggregate before joining.
* Use chunked CSV loading.
* For very large graphs, use `neo4j-admin database import`.

Memory-efficient design is critical due to the combinatorial growth of joins across arms, outcomes, and ontology mappings.

---

## Data Model Design Principles

1. Preserve relational structure instead of flattening.
2. Separate entities from attributes.
3. Maintain temporal information explicitly.
4. Model ontology hierarchy as graph relationships.
5. Enable many-to-many relationships without duplication.
6. Avoid Cartesian expansion during preprocessing.
7. Support predictive modeling features as graph properties.

---

## Potential Extensions

* Numeric outcome harmonization
* Full adverse event arm-level modeling
* Survival modeling for phase transition analysis
* Graph-based centrality and community detection
* Integration with DrugBank or PubMed knowledge graphs
* GraphRAG integration for LLM-driven querying

---

## Summary

ClinicalTrials.gov contains structured, longitudinal records of biomedical experimentation. By transforming this data into a normalized knowledge graph, this project enables structural, temporal, ontological, and quantitative reasoning over clinical research at scale.