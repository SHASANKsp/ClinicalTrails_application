# ClinicalTrials Knowledge Graph (CTKG) — Project Context File

> Paste this file at the start of a new chat to restore full project context.
> Last updated: end of session covering the full Streamlit app build + agent architecture.

---

## 1. Project Overview

A graph-native intelligence platform built on **Neo4j** that ingests raw ClinicalTrials.gov (AACT) data and exposes it through:
- Five analytics modules (Drug, Disease, Sponsor, Network, Geo/Time)
- A **GraphRAG** pipeline for grounded clinical Q&A
- A **Streamlit single-page app** with two side-by-side panels: Chat Agent + Analysis Agent
- All LLM calls use **Ollama** (local, free) — no paid API required

---

## 2. Repository File Structure

```
project/
├── injection.py                  # Data ingestion → Neo4j (MERGE statements, constraints)
├── application.py                # All analytics functions — returns (fig, df), no disk writes
├── network_graph_supplementary.py# Advanced graph visualisations (NetworkX-based)
├── streamlit_app.py              # Main Streamlit app — single page, two-panel layout
├── agentic_explainer.html        # Static HTML explainer of the architecture
└── CTKG_application_manual.ipynb # Original notebook (source of analytics logic)
```

---

## 3. Graph Schema

### Node Labels

| Label | Key Properties |
|---|---|
| `Study` | `nct_id`, `phases`, `overall_status`, `enrollment`, `start_date`, `completion_date`, `study_type`, `allocation`, `masking` |
| `Intervention` | `name`, `type` |
| `Condition` | `name` |
| `Sponsor` | `name`, `class` (Industry / Academic / Government) |
| `Arm` | `type` |
| `Location` | `country`, `city` |

### Relationships

| Relationship | Direction |
|---|---|
| `SPONSORS` | Sponsor → Study |
| `STUDIES` | Study → Condition |
| `USES_INTERVENTION` | Study → Intervention |
| `HAS_ARM` | Study → Arm |
| `CONDUCTED_AT` | Study → Location |

---

## 4. Neo4j Connection

Default credentials used in the app:
```
URI:      neo4j://localhost:7687
User:     neo4j
Password: CTrail@123
```

`KGClient` class in `application.py` wraps the driver. `test_connection()` runs `RETURN 1`.

---

## 5. application.py — Analytics Functions

All functions return `(fig, df)` where `fig` is a matplotlib Figure (or None for data-only) and `df` is a pandas DataFrame. No files are written to disk — compatible with Streamlit's `st.pyplot()`.

### Drug Intelligence
| Function | Description |
|---|---|
| `drug_evidence(kg, drug)` | Phase distribution, statuses, enrollment, conditions, sponsors |
| `drug_competition(kg, drug)` | Competitors in same conditions, ranked by trial count |
| `drug_geo(kg, drug)` | Countries running trials for a drug |
| `drug_paths(kg, drug)` | Multi-hop Drug→Trial→Condition→Sponsor paths — used for GraphRAG |

### Disease Analytics
| Function | Description |
|---|---|
| `disease_landscape(kg, disease)` | All drugs trialled, ranked by volume |
| `disease_phase_progression(kg, disease)` | Phase distribution |
| `disease_enrollment(kg, disease)` | Enrollment size histogram + median by phase |
| `disease_design(kg, disease)` | Allocation, masking, arm type breakdown |
| `disease_sponsor_diversity(kg, disease)` | Industry vs Academic vs Government split |

### Sponsor Intelligence
| Function | Description |
|---|---|
| `sponsor_portfolio(kg, sponsor)` | Condition areas the sponsor is active in |
| `sponsor_drugs(kg, sponsor)` | Drugs/interventions the sponsor is testing |
| `sponsor_pipeline(kg, sponsor)` | Phase × Status breakdown |
| `sponsor_geo(kg, sponsor)` | Geographic reach |
| `sponsor_collaborators(kg, sponsor)` | Co-sponsors in shared trials |

### Network Analytics
| Function | Description |
|---|---|
| `centrality(kg)` | Degree + betweenness centrality on drug–condition graph |
| `community_detection(kg)` | Louvain communities (requires `python-louvain`) |
| `drug_repurposing(kg)` | Drugs spanning most distinct conditions (scatter) |

### Geo / Temporal
| Function | Description |
|---|---|
| `trial_density(kg)` | Trial counts by country |
| `trial_timeline(kg)` | Monthly trial start activity |
| `geo_phase_heatmap(kg)` | Country × Phase heatmap |

### GraphRAG Context
```python
build_graphrag_context(kg, drug=None, disease=None, sponsor=None, limit=150)
```
Assembles multi-hop paths as structured text for LLM grounding.

---

## 6. network_graph_supplementary.py — Functions

| Function | Description |
|---|---|
| `drug_condition_subgraph(kg, drug)` | Ego-graph centred on a drug node |
| `sponsor_network(kg, min_shared_trials)` | Weighted sponsor co-occurrence network |
| `condition_similarity_network(kg, min_shared)` | Conditions linked by shared drugs |
| `graph_metrics(kg)` | Summary table: nodes, edges, density, components |
| `degree_distribution(kg)` | Log-scale + log-log degree distribution plots |
| `bridge_drugs(kg)` | Betweenness centrality — structural bridge nodes |

---

## 7. streamlit_app.py — Architecture

### Layout
Single page, two columns side-by-side:
- **Left (5 units):** Chat Agent
- **Right (7 units):** Analysis Agent — Plots & Summaries

### Session State Keys
```python
st.session_state.kg             # KGClient instance
st.session_state.connected      # bool
st.session_state.ollama_models  # list of detected model names
st.session_state.chat_history   # list of {role, content, meta}
st.session_state.analysis_panels # list of {turn_id, label, panels:[...]}
st.session_state.conv_context   # {entity_type, entity_value, search_token}
```

### Sidebar Controls
- Neo4j URI / user / password + Connect button
- Ollama URL + Detect Models button + model selector
- Clear Session button (resets chat_history, analysis_panels, conv_context)

### Chat Panel Features
- Checkbox: **"Generate plots and summaries for this query"** (default: unchecked)
  - When unchecked: text-only answer, no KG queries for plots
  - When checked: full analysis pipeline runs, results appended to analysis_panels
- Each assistant message has a collapsible **Query Details** expander showing:
  - Entity type, canonical name, search token
  - Resolution method
  - Analyses triggered
  - Exact Cypher queries + parameters
  - Model used, context size

### Analysis Panel Features
- Renders all turns in order (oldest first = Turn 1, Turn 2, ...)
- Each turn shown as a styled heading (no nested expanders)
- Each panel shows: chart → LLM summary → "Data & Query" expander (top-level)
- LLM summaries are cached on `panel["summary"]` — not regenerated on rerun
- Figures stored as matplotlib objects — not closed after render (persist across reruns)

---

## 8. The Query Pipeline (Most Important)

```
User question
    │
    ├─ NCT ID regex \bNCT\d{8}\b → short-circuit → nct_detail Cypher → KG context
    │
    └─ llm_extract_entity()          ← Ollama call, temp=0, num_predict=80
           │                           Returns {entity_type, entity_value} or null
           │
           ├─ null → hard stop, user warned, no defaults used
           │
           └─ entity found
                 │
                 ├─ network/geo → accepted directly (no KG confirmation needed)
                 │
                 └─ drug/disease/sponsor
                       │
                       └─ _kg_confirm()     ← Cypher CONTAINS lookup in Neo4j
                             │               Returns canonical_name + search_token
                             │               search_token = original LLM value
                             │               (catches all branches: "amgen" → all Amgen entities)
                             │
                             └─ conv_context carryover check
                                   │ No new entity + prior context → inherit
                                   │ Pronoun/reference → force prior entity
                                   │
                                   └─ llm_select_analyses()  ← Ollama call, temp=0
                                         │  Closed menu per entity type
                                         │  Validates keys against ANALYSIS_CATALOGUE
                                         │
                                         └─ run_analyses()   ← Cypher execution
                                               │  Uses search_token for CONTAINS
                                               │  Returns (fig, df) per analysis
                                               │
                                               ├─ build_graphrag_context() → KG text
                                               │
                                               └─ Ollama answer stream
                                                     Full history + entity context
                                                     + KG evidence + data summary
                                                     No token limit
```

---

## 9. Key Functions in streamlit_app.py

### `llm_extract_entity(question, ollama_url, ollama_model, conversation_history)`
- Replaces all former regex/stopword/token-extraction logic
- Single Ollama call, `temperature=0`, `num_predict=80`, 20s timeout
- Returns `{entity_type, entity_value}` or `None`
- Includes recent conversation history for follow-up resolution

### `_kg_confirm(kg, entity_value, entity_type)`
- Tries the LLM-suggested type first, falls through to others
- Prefers exact match, otherwise shortest (most specific) name
- Returns `{entity_type, entity_value (canonical), search_token (original)}`

### `detect_intent(question, kg, ollama_url, ollama_model, conversation_history)`
- Orchestrates the full pipeline above
- Returns `{entity_type, entity_value, search_token, analyses=None, resolution}`
- `analyses=None` is a sentinel — caller must call `llm_select_analyses()`
- Resolution values: `"nct_id"` | `"llm_kg"` | `"llm_only"` | `"keyword"` | `"fallback"`

### `llm_select_analyses(question, entity_type, entity_value, ollama_url, ollama_model, conversation_history)`
- Closed vocabulary: only keys in `ANALYSIS_CATALOGUE` can be returned
- Menu is pre-filtered to `CATALOGUE_BY_TYPE[entity_type]` (5 options max)
- Hallucinated keys silently dropped
- Falls back to all analyses for entity type on failure

### `run_analyses(kg, intent, defaults)`
- `search_token` used in all Cypher CONTAINS — not canonical name
- `drug_paths` always appended for drug queries (needed for GraphRAG)
- NCT detail handled via `query_nct_detail(kg, nct_id)`
- `_add(key, title, fig, df)` — 4-arg only (no params arg)

---

## 10. ANALYSIS_CATALOGUE — Full List

```python
ANALYSIS_CATALOGUE = {
    # Drug
    "drug_evidence":             "Phase distribution, trial statuses, enrollment...",
    "drug_competition":          "Other drugs tested in same conditions...",
    "drug_geo":                  "Countries where trials are conducted...",
    "drug_paths":                "Multi-hop paths for GraphRAG context",
    # Disease
    "disease_landscape":         "All drugs trialled for a disease...",
    "disease_phase":             "Phase distribution across trials...",
    "disease_enrollment":        "Enrollment size distribution...",
    "disease_design":            "Trial design breakdown...",
    "disease_sponsor_diversity": "Sponsor mix by class...",
    # Sponsor
    "sponsor_portfolio":         "Conditions sponsor is active in...",
    "sponsor_drugs":             "Drugs sponsor is testing...",
    "sponsor_pipeline":          "Phase × Status breakdown...",
    "sponsor_geo":               "Geographic reach...",
    "sponsor_collaborators":     "Co-sponsors in shared trials...",
    # Network
    "centrality":                "Degree + betweenness centrality...",
    "repurposing":               "Drugs spanning most conditions...",
    # Geo
    "geo_density":               "Trial counts by country...",
    "trial_timeline":            "Monthly trial start activity...",
    # Special
    "nct_detail":                "Full trial record by NCT ID",
}
```

---

## 11. CYPHER_REGISTRY — Key Queries

All queries accessible via `CYPHER_REGISTRY["key"]` for display in Query Details expander.

Key parameters:
- Drug queries: `$drug` (search token, lowercase CONTAINS)
- Disease queries: `$disease`
- Sponsor queries: `$sponsor`
- NCT queries: `$nct_id` (exact match on `{nct_id: $nct_id}`)

---

## 12. Known Bugs Fixed in This Session

| Bug | Fix |
|---|---|
| `"explain"` matched `"PNE-explained group"` via CONTAINS | Replaced entire token/stopword approach with `llm_extract_entity()` |
| "amgen" → only one branch matched | `search_token` = original LLM value; all Cypher uses CONTAINS on it |
| "Check-Cap Ltd." spurious match on "branches" | Same fix — LLM extracts "amgen" directly |
| `analyses` always same 4 for any sponsor query | `llm_select_analyses()` picks based on question semantics |
| `_add()` called with 5 args (exception handler) | Fixed to 4-arg call |
| Nested `st.expander` crash | Outer turn expander replaced with styled `st.markdown` div |
| `conv_context` KeyError on fresh start | Initialised in session state `defaults` dict |
| Analysis panels overwritten each query | Changed to append per-turn dict; analysis_panels is a list of turns |
| Turn order reversed (newest first) | Changed to `for turn in all_turns:` (oldest first) |
| No token limit on LLM answer | Ollama called with `"num_predict": -1` |
| Hardcoded Semaglutide/Diabetes/Novo Nordisk defaults | Removed; hard stop fires when entity unresolved |
| NCT ID queries misrouted to entity pipeline | NCT regex short-circuit added at top of `detect_intent` |
| `drug_paths` always hardcoded | `drug_paths` appended for drug queries; `sponsor_drugs` handler added |
| Plots regenerated on every Streamlit rerun | LLM summaries cached on `panel["summary"]`; figures stored in session state |
| `conv_context` overwritten by NCT queries | `if intent["entity_type"] != "nct": update conv_context` |

---

## 13. What Is and Is Not Agentic

**Current system is:** LLM-orchestrated analytical pipeline with GraphRAG.
- LLM performs entity extraction and analysis selection
- Python executes all tools deterministically
- Not agentic: LLM cannot decide to run additional queries based on results

**To make it agentic would require:**
1. Tool-calling loop — LLM calls `query_neo4j(cypher)` directly and iterates
2. Reflection/scratchpad — LLM reasons about what it found before deciding next step
3. Multi-step planning — decompose complex queries into sub-goals

---

## 14. Dependencies

```bash
pip install streamlit neo4j networkx pandas matplotlib seaborn requests python-louvain
```

Ollama must be running locally:
```bash
ollama serve
ollama pull llama3   # or mistral, gemma3, phi4, etc.
```

Run app:
```bash
streamlit run streamlit_app.py
```

---

## 15. Files Produced in This Session

| File | Status |
|---|---|
| `streamlit_app.py` | Main deliverable — 1,462 lines |
| `application.py` | Analytics library — 797 lines |
| `network_graph_supplementary.py` | Graph visualisations — 374 lines |
| `agentic_explainer.html` | Architecture explainer page |

---

## 16. Suggested Next Steps

1. **True agentic upgrade** — implement tool-calling loop so LLM can request additional Cypher queries based on what it finds
2. **Vector similarity fallback** — if `_kg_confirm()` finds no match, use `rapidfuzz` fuzzy matching against a pre-loaded entity list
3. **Multi-entity queries** — "compare Amgen and Pfizer" requires parallel resolution of two entities
4. **Export** — add download buttons for DataFrames as CSV and figures as PNG
5. **Auth** — add Streamlit login if deploying to a shared server
