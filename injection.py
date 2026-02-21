from neo4j import GraphDatabase
import pandas as pd
import os
import math
import time

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "CTrail@123"

DATA_DIR = "kg_output"
BATCH_SIZE = 5000

# Required merge keys per label
REQUIRED_KEYS = {
    "Studies": ["nct_id"],
    "Sponsors": ["lead_sponsor", "nct_id"],
    "Conditions": ["condition", "nct_id"],
    "Interventions": ["intervention_name", "nct_id"],
    "Arms": ["id", "nct_id"],
    "Locations": ["id", "nct_id"],
    "Outcomes": ["measure", "nct_id"],
    "AdverseEvents": ["nct_id"]
}

# =============================
# Connect
# =============================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


def run_query(query, parameters=None):
    with driver.session() as session:
        session.run(query, parameters or {})


# =============================
# Utility: Batch Loader
# =============================

def batch_loader(df, query, label):

    total = len(df)
    batches = math.ceil(total / BATCH_SIZE)

    print(f"\n--- Loading {label} ---")
    print(f"Total rows: {total}")
    print(f"Batches: {batches} (Batch size: {BATCH_SIZE})")

    required = REQUIRED_KEYS.get(label, [])

    start_time = time.time()

    for i in range(batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, total)

        batch_df = df.iloc[start:end].copy()

        # Convert NaN → None
        batch_df = batch_df.where(pd.notnull(batch_df), None)

        # Strip whitespace from string columns
        for col in batch_df.columns:
            if batch_df[col].dtype == object:
                batch_df[col] = batch_df[col].apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )

        # Drop rows missing required merge keys
        for col in required:
            batch_df = batch_df[batch_df[col].notnull()]

        if len(batch_df) == 0:
            print(f"{label}: Batch {i+1}/{batches} skipped (no valid rows)")
            continue

        run_query(query, {"rows": batch_df.to_dict("records")})

        print(f"{label}: Batch {i+1}/{batches} loaded ({end}/{total})")

    elapsed = time.time() - start_time
    print(f"{label} completed in {elapsed:.2f} seconds.")


# =============================
# Constraints
# =============================

def create_constraints():
    print("\nCreating constraints...")

    queries = [
        "CREATE CONSTRAINT study_id IF NOT EXISTS FOR (s:Study) REQUIRE s.nct_id IS UNIQUE",
        "CREATE CONSTRAINT intervention_name IF NOT EXISTS FOR (i:Intervention) REQUIRE i.name IS UNIQUE",
        "CREATE CONSTRAINT condition_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT sponsor_name IF NOT EXISTS FOR (sp:Sponsor) REQUIRE sp.name IS UNIQUE",
        "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
        "CREATE CONSTRAINT arm_id IF NOT EXISTS FOR (a:Arm) REQUIRE a.id IS UNIQUE",
    ]

    for q in queries:
        run_query(q)

    print("Constraints ready.")


# =============================
# Loaders
# =============================

def load_studies():
    df = pd.read_csv(os.path.join(DATA_DIR, "studies.csv"))

    query = """
    UNWIND $rows AS row
    MERGE (s:Study {nct_id: row.nct_id})
    SET s += row
    """

    batch_loader(df, query, "Studies")


def load_sponsors():
    df = pd.read_csv(os.path.join(DATA_DIR, "sponsors.csv"))

    query = """
    UNWIND $rows AS row
    MERGE (sp:Sponsor {name: row.lead_sponsor})
    SET sp.class = row.lead_sponsor_class
    WITH sp, row
    MATCH (s:Study {nct_id: row.nct_id})
    MERGE (sp)-[:SPONSORS]->(s)
    """

    batch_loader(df, query, "Sponsors")


def load_conditions():
    df = pd.read_csv(os.path.join(DATA_DIR, "conditions.csv"))

    query = """
    UNWIND $rows AS row
    MERGE (c:Condition {name: row.condition})
    WITH c, row
    MATCH (s:Study {nct_id: row.nct_id})
    MERGE (s)-[:STUDIES]->(c)
    """

    batch_loader(df, query, "Conditions")


def load_interventions():
    df = pd.read_csv(os.path.join(DATA_DIR, "interventions.csv"))

    query = """
    UNWIND $rows AS row
    MERGE (i:Intervention {name: row.intervention_name})
    SET i.type = row.intervention_type,
        i.description = row.description
    WITH i, row
    MATCH (s:Study {nct_id: row.nct_id})
    MERGE (s)-[:USES_INTERVENTION]->(i)
    """

    batch_loader(df, query, "Interventions")


def load_arms():
    df = pd.read_csv(os.path.join(DATA_DIR, "arms.csv"))
    df["id"] = df["nct_id"] + "_" + df["arm_label"].astype(str)

    query = """
    UNWIND $rows AS row
    MERGE (a:Arm {id: row.id})
    SET a.label = row.arm_label,
        a.type = row.arm_type,
        a.description = row.arm_description
    WITH a, row
    MATCH (s:Study {nct_id: row.nct_id})
    MERGE (s)-[:HAS_ARM]->(a)
    """

    batch_loader(df, query, "Arms")


def load_locations():
    df = pd.read_csv(os.path.join(DATA_DIR, "locations.csv"))
    df["id"] = df["nct_id"] + "_" + df["facility"].astype(str)

    query = """
    UNWIND $rows AS row
    MERGE (l:Location {id: row.id})
    SET l.facility = row.facility,
        l.city = row.city,
        l.country = row.country,
        l.lat = row.lat,
        l.lon = row.lon
    WITH l, row
    MATCH (s:Study {nct_id: row.nct_id})
    MERGE (s)-[:CONDUCTED_AT]->(l)
    """

    batch_loader(df, query, "Locations")


def load_outcomes():
    df = pd.read_csv(os.path.join(DATA_DIR, "outcomes.csv"))

    query = """
    UNWIND $rows AS row
    MERGE (o:Outcome {measure: row.measure})
    SET o.timeframe = row.timeframe,
        o.type = row.type
    WITH o, row
    MATCH (s:Study {nct_id: row.nct_id})
    MERGE (s)-[:MEASURES]->(o)
    """

    batch_loader(df, query, "Outcomes")


def load_adverse_events():
    df = pd.read_csv(os.path.join(DATA_DIR, "adverse_events.csv"))

    query = """
    UNWIND $rows AS row
    CREATE (ae:AdverseEvent {
        term: row.event_term,
        num_affected: row.num_affected,
        num_at_risk: row.num_at_risk
    })
    WITH ae, row
    MATCH (s:Study {nct_id: row.nct_id})
    MERGE (s)-[:REPORTS_EVENT]->(ae)
    """

    batch_loader(df, query, "AdverseEvents")


# =============================
# MAIN
# =============================

if __name__ == "__main__":

    print("\n=========================================")
    print("Starting ClinicalTrials KG Construction")
    print("=========================================")

    total_start = time.time()

    create_constraints()
    load_studies()
    load_sponsors()
    load_conditions()
    load_interventions()
    load_arms()
    load_locations()
    load_outcomes()

    driver.close()

    total_time = time.time() - total_start

    print("\n=========================================")
    print("Knowledge Graph Successfully Built")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("=========================================")
