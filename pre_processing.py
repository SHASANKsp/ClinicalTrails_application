#!/usr/bin/env python3
"""
split_json_to_edge_files_with_progress.py

Stream a ClinicalTrials JSONL/JSON (list) file and write per-node and per-relationship JSONL files
for later Neo4j ingestion. Uses tqdm progress bar.

Usage:
    python split_json_to_edge_files_with_progress.py --input clinical_trials_dump.jsonl[.gz] --outdir kg_staging

Notes:
 - For very large JSON array files (a single huge '[' ... ']'), the script will attempt to load the file into memory.
   This may be memory-heavy. Prefer JSONL (one JSON object per line) or gzipped JSONL.
"""

import json
import gzip
import argparse
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Iterable
from tqdm import tqdm

# --------------------------
# Config
# --------------------------
DEAD_LETTER_NAME = "dead_letter.jsonl"
LOG_FILE = "split_etl_progress.log"

# --------------------------
# Logging
# --------------------------
logging.basicConfig(level=logging.INFO, filename=LOG_FILE,
                    format="%(asctime)s %(levelname)s %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# --------------------------
# Utilities
# --------------------------
def open_input(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def safe_get(dct: Dict[str, Any], path: str, default=None):
    cur = dct
    for p in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def normalize_intervention_name(name: str):
    if not name:
        return name
    s = name.strip()
    for prefix in ["Drug:", "Other:", "Device:", "Procedure:", "Biological:"]:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    return s

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def write_jsonl(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

# --------------------------
# Writers factory (open many files)
# --------------------------
def open_writers(outdir: Path):
    ensure_dir(outdir)
    writers = {}
    def ow(name):
        p = outdir / name
        # open in append mode so multiple runs can be resumed if desired
        writers[name] = open(p, "a", encoding="utf-8")
        return writers[name]
    # Node files
    for fname in [
        "trials.jsonl","organizations.jsonl","conditions.jsonl","interventions.jsonl",
        "arms.jsonl","sites.jsonl","investigators.jsonl","contacts.jsonl",
        "outcomes.jsonl","results.jsonl","baseline_groups.jsonl","baseline_measures.jsonl",
        "participant_flow_groups.jsonl","adverse_events.jsonl","eligibility.jsonl",
        "publications.jsonl","versions.jsonl"
    ]:
        ow(fname)

    # Relationship files
    for fname in [
        "trial_sponsoredby_rel.jsonl","trial_studies_rel.jsonl","trial_uses_intervention_rel.jsonl",
        "trial_has_arm_rel.jsonl","arm_contains_intervention_rel.jsonl","trial_has_site_rel.jsonl",
        "trial_has_contact_rel.jsonl","trial_has_investigator_rel.jsonl","trial_has_outcome_rel.jsonl",
        "outcome_has_result_rel.jsonl","trial_has_version_rel.jsonl","participantflow_has_achievement_rel.jsonl",
        "adverseevent_has_stat_rel.jsonl"
    ]:
        ow(fname)
    return writers

# --------------------------
# Input counting helpers for progress
# --------------------------
def estimate_total_lines(path: Path):
    """
    Estimate total items for progress bar.
    If file is JSONL(.gz) we can count lines cheaply.
    If file begins with '[' (a JSON array), attempt to parse as JSON and return len(array).
      Warning: this loads the JSON array in memory and may be heavy for big files.
    If counting fails, return None and tqdm will use indeterminate mode.
    """
    try:
        # fast check first char
        with open_input(path) as fh:
            first_non_ws = None
            while True:
                ch = fh.read(1)
                if not ch:
                    break
                if not ch.isspace():
                    first_non_ws = ch
                    break
            fh.seek(0)
            if first_non_ws == "[":
                # JSON array: parse (may be heavy)
                logging.info("Input looks like a JSON array (starts with '['). Attempting to parse entire array to count elements (may use a lot of memory).")
                data = json.load(fh)
                count = len(data)
                logging.info(f"Counted {count} elements in JSON array.")
                return count
            else:
                # JSONL: count lines
                cnt = 0
                for _ in fh:
                    cnt += 1
                logging.info(f"Counted {cnt} lines in input JSONL.")
                return cnt
    except Exception as e:
        logging.exception("Failed to estimate total lines (will run without total).")
        return None

# --------------------------
# Extraction logic (single pass with progress)
# --------------------------
def process_file_with_progress(input_path: Path, outdir: Path):
    writers = open_writers(outdir)
    dead_letter_fp = open(outdir / DEAD_LETTER_NAME, "a", encoding="utf-8")

    # small in-script dedupe sets to reduce duplicate writes
    seen_orgs = set()
    seen_conditions = set()
    seen_interventions = set()
    seen_trials = set()

    total_est = estimate_total_lines(input_path)
    logging.info(f"Starting stream. Estimated total items: {total_est}")

    # counters for progress postfix
    counters = {
        "trials": 0, "orgs": 0, "conditions": 0, "interventions": 0, "arms": 0, "sites": 0, "contacts": 0,
        "investigators": 0, "outcomes": 0, "results": 0, "elig": 0, "pubs": 0
    }

    # open input and iterate
    with open_input(input_path) as fh:
        # determine mode: JSON array or JSONL
        first_non_ws = None
        while True:
            ch = fh.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break
        fh.seek(0)

        if first_non_ws == "[":
            # JSON array mode: load whole array (warning in docstring)
            logging.info("Detected JSON array format — loading into memory (may be heavy).")
            data = json.load(fh)
            iterator = iter(data)
            total = len(data)
            pbar = tqdm(iterator, total=total, desc="Processing trials", unit="trial")
        else:
            # JSONL mode: iterate line by line
            # create a generator that yields parsed JSON objects per line; use tqdm with known total if available
            def line_generator(f):
                for line in f:
                    if not line.strip():
                        continue
                    yield line
            line_iter = line_generator(fh)
            pbar = tqdm(line_iter, total=total_est, desc="Processing trials", unit="trial")

        processed = 0
        for raw_item in pbar:
            # parse raw_item (either str or dict)
            if isinstance(raw_item, str):
                line = raw_item.strip()
                try:
                    js = json.loads(line)
                except Exception as e:
                    logging.exception("JSON parse error")
                    write_jsonl(dead_letter_fp, {"error": "parse", "line": line[:400]})
                    continue
            else:
                # already a dict (when iterating over JSON array)
                js = raw_item

            try:
                ps = js.get("protocolSection", {})
                idm = ps.get("identificationModule", {})
                nctId = safe_get(idm, "nctId")
                if not nctId:
                    write_jsonl(dead_letter_fp, {"error": "missing_nct", "record_excerpt": str(js)[:400]})
                    continue

                # --- Trial node ---
                trial_node = {
                    "nctId": nctId,
                    "orgStudyId": safe_get(idm, "orgStudyIdInfo.id"),
                    "briefTitle": safe_get(idm, "briefTitle"),
                    "officialTitle": safe_get(idm, "officialTitle"),
                    "acronym": safe_get(idm, "acronym"),
                    "overallStatus": safe_get(ps, "statusModule.overallStatus"),
                    "statusVerifiedDate": safe_get(ps, "statusModule.statusVerifiedDate"),
                    "startDate": safe_get(ps, "statusModule.startDateStruct.date"),
                    "primaryCompletionDate": safe_get(ps, "statusModule.primaryCompletionDateStruct.date"),
                    "completionDate": safe_get(ps, "statusModule.completionDateStruct.date"),
                    "studyFirstSubmitDate": safe_get(ps, "statusModule.studyFirstSubmitDate"),
                    "studyFirstPostDate": safe_get(ps, "statusModule.studyFirstPostDateStruct.date"),
                    "lastUpdatePostDate": safe_get(ps, "statusModule.lastUpdatePostDateStruct.date"),
                    "studyType": safe_get(ps, "designModule.studyType"),
                    "phases": safe_get(ps, "designModule.phases"),
                    "interventionModel": safe_get(ps, "designModule.designInfo.interventionModel"),
                    "allocation": safe_get(ps, "designModule.designInfo.allocation"),
                    "primaryPurpose": safe_get(ps, "designModule.designInfo.primaryPurpose"),
                    "masking": safe_get(ps, "designModule.designInfo.maskingInfo.masking"),
                    "enrollmentCount": safe_get(ps, "designModule.enrollmentInfo.count"),
                    "enrollmentType": safe_get(ps, "designModule.enrollmentInfo.type"),
                    "briefSummary": safe_get(ps, "descriptionModule.briefSummary"),
                    "detailedDescription": safe_get(ps, "descriptionModule.detailedDescription"),
                    "hasResults": js.get("hasResults", False),
                    "ipdSharing": safe_get(ps, "ipdSharingStatementModule.ipdSharing"),
                    "versionHolder": safe_get(ps, "derivedSection.miscInfoModule.versionHolder"),
                    "rawJsonPath": f"{input_path}::{nctId}"
                }
                if nctId not in seen_trials:
                    write_jsonl(writers["trials.jsonl"], trial_node)
                    seen_trials.add(nctId)
                    counters["trials"] += 1

                # --- Organization (lead sponsor) ---
                lead_name = safe_get(ps, "sponsorCollaboratorsModule.leadSponsor.name")
                lead_class = safe_get(ps, "sponsorCollaboratorsModule.leadSponsor.class")
                if lead_name:
                    canon = lead_name.strip().lower()
                    if canon not in seen_orgs:
                        org_node = {"name": lead_name, "class": lead_class, "rawSourceField": "sponsorCollaboratorsModule.leadSponsor"}
                        write_jsonl(writers["organizations.jsonl"], org_node)
                        seen_orgs.add(canon)
                        counters["orgs"] += 1
                    # relationship record
                    rel = {"from_nct": nctId, "org_name": lead_name, "role": "lead_sponsor"}
                    write_jsonl(writers["trial_sponsoredby_rel.jsonl"], rel)

                # --- Conditions ---
                conds = safe_get(ps, "conditionsModule.conditions", []) or []
                for cond in conds:
                    nm = (cond or "").strip()
                    if not nm:
                        continue
                    if nm.lower() not in seen_conditions:
                        write_jsonl(writers["conditions.jsonl"], {"name": nm, "rawSource": "conditionsModule.conditions"})
                        seen_conditions.add(nm.lower())
                        counters["conditions"] += 1
                    write_jsonl(writers["trial_studies_rel.jsonl"], {"from_nct": nctId, "condition_name": nm})

                # --- Interventions ---
                ints = safe_get(ps, "armsInterventionsModule.interventions", []) or []
                for it in ints:
                    name = it.get("name")
                    name_norm = normalize_intervention_name(name)
                    if not name_norm:
                        continue
                    key = name_norm.lower()
                    if key not in seen_interventions:
                        int_node = {
                            "name": name_norm,
                            "type": it.get("type"),
                            "description": it.get("description"),
                            "otherNames": it.get("otherNames", []),
                            "rawSource": "armsInterventionsModule.interventions"
                        }
                        write_jsonl(writers["interventions.jsonl"], int_node)
                        seen_interventions.add(key)
                        counters["interventions"] += 1
                    # relation trial -> intervention
                    write_jsonl(writers["trial_uses_intervention_rel.jsonl"], {"from_nct": nctId, "intervention_name": name_norm})

                # --- Arms ---
                armGroups = safe_get(ps, "armsInterventionsModule.armGroups", []) or []
                for ag in armGroups:
                    arm_id = f"{nctId}::{ag.get('label')}"
                    arm_node = {
                        "armId": arm_id,
                        "label": ag.get("label"),
                        "type": ag.get("type"),
                        "description": ag.get("description"),
                        "interventionNames": ag.get("interventionNames", []),
                        "nctId": nctId
                    }
                    write_jsonl(writers["arms.jsonl"], arm_node)
                    write_jsonl(writers["trial_has_arm_rel.jsonl"], {"from_nct": nctId, "armId": arm_id})
                    counters["arms"] += 1
                    for iname in ag.get("interventionNames", []):
                        iname_norm = normalize_intervention_name(iname)
                        if iname_norm:
                            write_jsonl(writers["arm_contains_intervention_rel.jsonl"], {"armId": arm_id, "intervention_name": iname_norm})

                # --- Sites ---
                locs = safe_get(ps, "contactsLocationsModule.locations", []) or []
                for s in locs:
                    site_node = {
                        "facility": s.get("facility"),
                        "city": s.get("city"),
                        "state": s.get("state"),
                        "zip": s.get("zip"),
                        "country": s.get("country"),
                        "status": s.get("status"),
                        "latitude": s.get("geoPoint", {}).get("lat"),
                        "longitude": s.get("geoPoint", {}).get("lon")
                    }
                    write_jsonl(writers["sites.jsonl"], site_node)
                    write_jsonl(writers["trial_has_site_rel.jsonl"], {"from_nct": nctId, "facility": s.get("facility"), "city": s.get("city"), "country": s.get("country")})
                    counters["sites"] += 1

                # --- Contacts & Investigators ---
                contacts = safe_get(ps, "contactsLocationsModule.centralContacts", []) or []
                for c in contacts:
                    contact_node = {"name": c.get("name"), "role": c.get("role"), "phone": c.get("phone"), "email": c.get("email"), "nctId": nctId}
                    write_jsonl(writers["contacts.jsonl"], contact_node)
                    write_jsonl(writers["trial_has_contact_rel.jsonl"], {"from_nct": nctId, "contact_name": c.get("name")})
                    counters["contacts"] += 1
                officials = safe_get(ps, "contactsLocationsModule.overallOfficials", []) or []
                for of in officials:
                    inv_node = {"name": of.get("name"), "affiliation": of.get("affiliation"), "role": of.get("role"), "nctId": nctId}
                    write_jsonl(writers["investigators.jsonl"], inv_node)
                    write_jsonl(writers["trial_has_investigator_rel.jsonl"], {"from_nct": nctId, "investigator_name": of.get("name")})
                    counters["investigators"] += 1

                # --- Outcomes (primary/secondary/other) ---
                primary = safe_get(ps, "outcomesModule.primaryOutcomes", []) or []
                secondary = safe_get(ps, "outcomesModule.secondaryOutcomes", []) or []
                other_out = safe_get(ps, "outcomesModule.otherOutcomes", []) or []
                for o in (primary + secondary + other_out):
                    measure = o.get("measure") or o.get("title")
                    outcome_id = f"{nctId}::" + (measure or "unknown")[:120]
                    outcome_node = {"outcomeId": outcome_id, "nctId": nctId, "measure": measure, "description": o.get("description"), "timeFrame": o.get("timeFrame")}
                    write_jsonl(writers["outcomes.jsonl"], outcome_node)
                    write_jsonl(writers["trial_has_outcome_rel.jsonl"], {"from_nct": nctId, "outcomeId": outcome_id})
                    counters["outcomes"] += 1

                # --- resultsSection: create simple Result nodes where present ---
                rs = js.get("resultsSection", {}) or {}
                om = safe_get(rs, "outcomeMeasuresModule.outcomeMeasures", []) or []
                for omm in om:
                    classes = omm.get("classes", []) or []
                    for cls in classes:
                        cats = cls.get("categories", []) or []
                        for cat in cats:
                            meas_list = cat.get("measurements", []) or []
                            for m in meas_list:
                                res_id = f"{nctId}::{omm.get('title') or omm.get('type')}::{m.get('groupId')}"
                                result_node = {
                                    "resultId": res_id,
                                    "nctId": nctId,
                                    "outcomeTitle": omm.get("title"),
                                    "groupId": m.get("groupId"),
                                    "value": m.get("value"),
                                    "spread": m.get("spread")
                                }
                                write_jsonl(writers["results.jsonl"], result_node)
                                write_jsonl(writers["outcome_has_result_rel.jsonl"], {"outcomeTitle": omm.get("title"), "resultId": res_id})
                                counters["results"] += 1

                # --- participantFlowModule (groups + achievements) ---
                pf = safe_get(rs, "participantFlowModule", {}) or {}
                groups_pf = pf.get("groups", []) or []
                periods = pf.get("periods", []) or []
                for g in groups_pf:
                    pg_id = f"{nctId}::PF::{g.get('id') or g.get('title')}"
                    write_jsonl(writers["participant_flow_groups.jsonl"], {"flowGroupId": pg_id, "title": g.get("title"), "description": g.get("description"), "nctId": nctId})
                for p in periods:
                    milestones = p.get("milestones", []) or []
                    for ms in milestones:
                        achs = ms.get("achievements", []) or []
                        for a in achs:
                            write_jsonl(writers["participantflow_has_achievement_rel.jsonl"], {
                                "flowGroupId": f"{nctId}::PF::{a.get('groupId')}",
                                "periodTitle": p.get("title"),
                                "type": ms.get("type"),
                                "numSubjects": a.get("numSubjects"),
                                "nctId": nctId
                            })

                # --- baseline characteristics ---
                bc = safe_get(rs, "baselineCharacteristicsModule", {}) or {}
                groups_bc = bc.get("groups", []) or []
                measures = bc.get("measures", []) or []
                for g in groups_bc:
                    bgid = f"{nctId}::BG::{g.get('title') or g.get('id')}"
                    write_jsonl(writers["baseline_groups.jsonl"], {"baselineGroupId": bgid, "title": g.get("title"), "description": g.get("description"), "nctId": nctId})
                for m in measures:
                    bm_id = f"{nctId}::BM::{m.get('title')}"
                    write_jsonl(writers["baseline_measures.jsonl"], {"baselineMeasureId": bm_id, "title": m.get("title"), "paramType": m.get("paramType"), "unitOfMeasure": m.get("unitOfMeasure"), "nctId": nctId, "raw": m})

                # --- adverse events ---
                ae = safe_get(rs, "adverseEventsModule", {}) or {}
                eventGroups = ae.get("eventGroups", []) or []
                for eg in eventGroups:
                    write_jsonl(writers["adverse_events.jsonl"], {"nctId": nctId, "groupId": eg.get("id"), "title": eg.get("title"), "seriousNumAffected": eg.get("seriousNumAffected"), "otherNumAffected": eg.get("otherNumAffected")})
                seriousEvents = ae.get("seriousEvents", []) or []
                for se in seriousEvents:
                    term = se.get("term")
                    aev_id = f"{nctId}::AE::{term}"
                    write_jsonl(writers["adverse_events.jsonl"], {"adEventId": aev_id, "term": term, "organSystem": se.get("organSystem"), "assessmentType": se.get("assessmentType"), "nctId": nctId})
                    stats = se.get("stats", []) or []
                    for st in stats:
                        write_jsonl(writers["adverseevent_has_stat_rel.jsonl"], {"adEventId": aev_id, "groupId": st.get("groupId"), "numEvents": st.get("numEvents"), "numAffected": st.get("numAffected"), "numAtRisk": st.get("numAtRisk"), "nctId": nctId})

                # --- eligibility parsing (naive) ---
                elig_text = safe_get(ps, "eligibilityModule.eligibilityCriteria")
                if elig_text:
                    inc = []
                    exc = []
                    lower = elig_text.lower()
                    if "inclusion criteria" in lower or "exclusion criteria" in lower:
                        lines = [l.strip() for l in elig_text.splitlines() if l.strip()]
                        mode = None
                        for L in lines:
                            ll = L.lower()
                            if "inclusion" in ll and "criteria" in ll:
                                mode = "INCLUSION"
                                continue
                            if "exclusion" in ll and "criteria" in ll:
                                mode = "EXCLUSION"
                                continue
                            if mode == "INCLUSION":
                                inc.append(L)
                            elif mode == "EXCLUSION":
                                exc.append(L)
                            else:
                                if L.startswith("*") or L.startswith("-"):
                                    inc.append(L.lstrip("*- ").strip())
                                else:
                                    inc.append(L)
                    else:
                        inc = [elig_text]
                    seq = 0
                    for t in inc:
                        write_jsonl(writers["eligibility.jsonl"], {"criterionId": f"{nctId}::IN::#{seq}", "nctId": nctId, "type": "INCLUSION", "text": t, "sequence": seq})
                        seq += 1
                        counters["elig"] += 1
                    seq = 0
                    for t in exc:
                        write_jsonl(writers["eligibility.jsonl"], {"criterionId": f"{nctId}::EX::#{seq}", "nctId": nctId, "type": "EXCLUSION", "text": t, "sequence": seq})
                        seq += 1
                        counters["elig"] += 1

                # --- publications / references ---
                refs = safe_get(ps, "referencesModule.references", []) or []
                for r in refs:
                    pmid = r.get("pmid")
                    pubid = pmid or f"{nctId}::REF::{r.get('type') or 'other'}"
                    write_jsonl(writers["publications.jsonl"], {"publicationId": pubid, "pmid": pmid, "citation": r.get("citation"), "type": r.get("type"), "nctId": nctId})
                    write_jsonl(writers["trial_has_contact_rel.jsonl"], {"from_nct": nctId, "publicationId": pubid})
                    counters["pubs"] += 1

                # --- version summary ---
                version_holder = safe_get(ps, "derivedSection.miscInfoModule.versionHolder")
                if version_holder:
                    write_jsonl(writers["versions.jsonl"], {"versionId": version_holder, "nctId": nctId})

            except Exception as e:
                logging.exception("Error processing trial")
                write_jsonl(dead_letter_fp, {"error": str(e), "nctId": nctId if 'nctId' in locals() else None, "excerpt": str(js)[:400]})

            processed += 1
            # update progress postfix with counters
            pbar.set_postfix({
                "trials": counters["trials"], "orgs": counters["orgs"], "ints": counters["interventions"], "arms": counters["arms"],
                "outcomes": counters["outcomes"], "results": counters["results"], "elig": counters["elig"]
            })

    # close writers
    for fp in writers.values():
        fp.close()
    dead_letter_fp.close()
    logging.info(f"Done. Processed ~{processed} records. Output dir: {outdir}")
    print("Summary counts:", counters)

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Split ClinicalTrials JSON into per-edge JSONL files (with progress)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file (or .json/.jsonl.gz) with one trial per line or JSON array")
    parser.add_argument("--outdir", "-o", type=str, default="kg_staging", help="Output staging directory")
    args = parser.parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    if not input_path.exists():
        logging.error(f"Input file {input_path} not found")
        raise SystemExit(1)
    logging.info(f"Starting split for {input_path} -> {outdir}")
    process_file_with_progress(input_path, outdir)
    logging.info("Finished split.")

if __name__ == "__main__":
    main()
