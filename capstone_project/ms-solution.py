# ------------------------------------------------------------
# File: claim_react_agent.py  (you can paste cells into code.ipynb)
# ------------------------------------------------------------
import os, json, re, uuid
from datetime import date, datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate

# -------------------------
# 0) Config & Data Loading
# -------------------------

DATA_DIR = "./"  # adjust if needed
REF_CODES_PATH = os.path.join(DATA_DIR, "reference_codes.json")
POLICIES_PATH = os.path.join(DATA_DIR, "insurance_policies.json")
VAL_RECORDS = os.path.join(DATA_DIR, "validation_records.json")
TEST_RECORDS = os.path.join(DATA_DIR, "test_records.json")


# If you don't have files yet, the code will still load if you point to your dataset.
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


reference_codes = load_json(REF_CODES_PATH)  # {"cpt": {...}, "icd10": {...}}
policies = load_json(
    POLICIES_PATH
)  # list[policy], each policy has "policy_id", "covered_procedures", etc.

# index policies by id for quick access
policy_by_id = {p["policy_id"]: p for p in policies}

# -------------------------
# 1) Utility: Compute Age
# -------------------------


def compute_age(dob_str: str, dos_str: str) -> int:
    """
    Compute age as of date_of_service:
    - age = completed years
    - inclusive lower, exclusive upper is handled later in the policy logic
    """
    dob = datetime.fromisoformat(dob_str).date()
    dos = datetime.fromisoformat(dos_str).date()
    years = dos.year - dob.year - ((dos.month, dos.day) < (dob.month, dob.day))
    return years


# Batch precompute age and normalize fields in a record (mutates a copy)
def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    # Prefer canonical field names the spec uses
    if "claim_amount" in out and "billed_amount" not in out:
        out["billed_amount"] = out["claim_amount"]
    # Compute age if missing
    if "age" not in out and "date_of_birth" in out and "date_of_service" in out:
        out["age"] = compute_age(out["date_of_birth"], out["date_of_service"])
    return out


# -------------------------
# 2) Tools (exactly three)
# -------------------------


# 2.1 summarize_patient_record(record_str)
@tool
def summarize_patient_record(record_str: str) -> str:
    """
    Extract a structured summary of a patient's claim record.
    Input: record_str (JSON string or plain text). Output: 7 labeled sections in order:
    - Patient Demographics (name, gender, age)
    - Insurance Policy ID
    - Diagnoses and Descriptions (ICD-10)
    - Procedures and Descriptions (CPT)
    - Preauthorization Status
    - Billed Amount (in USD)
    - Date of Service
    """
    # Parse JSON if provided
    try:
        rec = json.loads(record_str)
    except Exception:
        rec = {}

    # enrich code descriptions
    icd_map = reference_codes.get("icd10", {})
    cpt_map = reference_codes.get("cpt", {})

    name = rec.get("name", "Unknown")
    gender = rec.get("gender", "Unknown")
    age = rec.get("age", "Unknown")
    policy = rec.get("insurance_policy_id", "Unknown")
    diag_codes = rec.get("diagnosis_codes", [])
    proc_codes = rec.get("procedure_codes", [])
    preauth_req = rec.get(
        "preauthorization_required",
        rec.get(
            "preauthorization_required ", rec.get("preauthorization_required", False)
        ),
    )
    preauth_got = rec.get(
        "preauthorization_obtained",
        rec.get(
            "preauthorization_obtained ", rec.get("preauthorization_obtained", False)
        ),
    )
    billed = rec.get("billed_amount", rec.get("claim_amount", "Unknown"))
    dos = rec.get("date_of_service", "Unknown")

    diag_lines = []
    for d in diag_codes:
        desc = icd_map.get(d, "Unknown diagnosis")
        diag_lines.append(f"- {d}: {desc}")

    proc_lines = []
    for c in proc_codes:
        desc = cpt_map.get(c, "Unknown procedure")
        proc_lines.append(f"- {c}: {desc}")

    summary = f"""Patient Demographics:
- Name: {name}
- Gender: {gender}
- Age: {age}

Insurance Policy ID:
- {policy}

Diagnoses and Descriptions:
{chr(10).join(diag_lines) if diag_lines else "- None"}

Procedures and Descriptions:
{chr(10).join(proc_lines) if proc_lines else "- None"}

Preauthorization Status:
- Required: {bool(preauth_req)}
- Obtained: {bool(preauth_got)}

Billed Amount (in USD):
- {billed}

Date of Service:
- {dos}
"""
    return summary


# 2.2 summarize_policy_guideline(policy_id)
@tool
def summarize_policy_guideline(policy_id: str) -> str:
    """
    Summarize a policy by policy_id with sections:
    - Policy Details (policy ID, plan name)
    - Covered Procedures: for each -> Procedure Code & Description, Covered Diagnoses & Descriptions,
      Gender Restriction, Age Range, Preauthorization Requirement, Notes on Coverage
    """
    icd_map = reference_codes.get("icd10", {})
    cpt_map = reference_codes.get("cpt", {})

    policy = policy_by_id.get(policy_id)
    if not policy:
        return f"Policy Details:\n- Policy ID: {policy_id}\n- Plan Name: Unknown\n\nCovered Procedures:\n- None found"

    header = f"""Policy Details:
- Policy ID: {policy["policy_id"]}
- Plan Name: {policy.get("plan_name", "Unknown")}

Covered Procedures:"""

    lines = [header]
    for proc in policy.get("covered_procedures", []):
        code = proc.get("procedure_code")
        code_desc = cpt_map.get(code, "Unknown procedure")
        diag_list = proc.get("covered_diagnoses", [])
        diag_lines = []
        for d in diag_list:
            diag_lines.append(f"  - {d}: {icd_map.get(d, 'Unknown diagnosis')}")

        gender = proc.get("gender", "Any")
        age_rng = proc.get("age_range", [])
        if len(age_rng) == 2:
            age_text = f"[{age_rng[0]}, {age_rng[1]})"
        else:
            age_text = "Unknown"

        preauth = proc.get("requires_preauthorization", False)
        notes = proc.get("notes", "None")

        lines.append(
            f"""
- Procedure Code and Description:
  - {code}: {code_desc}
- Covered Diagnoses and Descriptions:
{chr(10).join(diag_lines) if diag_lines else "  - None"}
- Gender Restriction:
  - {gender}
- Age Range:
  - {age_text}
- Preauthorization Requirement:
  - {bool(preauth)}
- Notes on Coverage:
  - {notes}
"""
        )
    return "\n".join(lines)


# 2.3 check_claim_coverage(record_summary, policy_summary)
@tool
def check_claim_coverage(record_summary: str, policy_summary: str) -> str:
    """
    Determine coverage eligibility for the (single) claimed procedure using LLM reasoning.
    Output sections:
    - Coverage Review
    - Summary of Findings
    - Final Decision: APPROVE or ROUTE FOR REVIEW (with brief explanation)
    """
    # We delegate evaluation to the LLM but constrain format & logic via a prompt.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = PromptTemplate(
        template=(
            "You are a meticulous claims analyst. Evaluate claim coverage using the following:\n\n"
            "=== PATIENT RECORD SUMMARY ===\n{record_summary}\n\n"
            "=== POLICY SUMMARY ===\n{policy_summary}\n\n"
            "Rules:\n"
            "1) Approve only if ALL conditions are satisfied: (a) patient diag code(s) ∈ covered diagnoses for the claimed procedure; "
            "(b) procedure code is explicitly listed; (c) patient's age within policy age range [lower, upper) "
            "(inclusive lower bound, exclusive upper bound); (d) patient's gender matches; "
            "(e) if preauthorization required, it was obtained.\n"
            "2) Only evaluate the single procedure present in the patient record.\n"
            "3) Output MUST have exactly 3 sections in order and in plain text:\n"
            "   Coverage Review:\n"
            "   Summary of Findings:\n"
            "   Final Decision: APPROVE | ROUTE FOR REVIEW — <one-sentence reason>\n"
        ),
        input_variables=["record_summary", "policy_summary"],
    )
    chain = prompt | llm
    resp = chain.invoke(
        {"record_summary": record_summary, "policy_summary": policy_summary}
    )
    return resp.content


# -------------------------
# 3) Agent system prompt
# -------------------------

SYSTEM_PROMPT = """You are a ReAct-style, policy-compliant Insurance Claim Approval Agent.

TOOLS YOU MAY USE (and only these):
1) summarize_patient_record(record_str)
2) summarize_policy_guideline(policy_id)
3) check_claim_coverage(record_summary, policy_summary)

MANDATORY WORKFLOW (strict order):
Step A: Call summarize_patient_record(record_str) on the raw patient JSON/text.
Step B: Call summarize_policy_guideline(policy_id) using the policy ID from Step A.
Step C: Call check_claim_coverage(record_summary, policy_summary) using the outputs of Steps A and B.
Then produce the FINAL SINGLE-LINE OUTPUT:
Decision: APPROVE | ROUTE FOR REVIEW
Reason: <concise reason referencing the specific policy conditions (diagnosis, procedure, age [lower, upper), gender, preauthorization) that determined the outcome.>

Never skip a step. Never output Decision/Reason until Step C completes. Use exactly the three tools above and in this sequence.
"""

# -------------------------
# 4) Create ReAct Agent
# -------------------------

tools = [summarize_patient_record, summarize_policy_guideline, check_claim_coverage]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    state_modifier=SYSTEM_PROMPT,  # system instruction
    debug=True,
)

# -------------------------
# 5) Helper: Run agent for a single record
# -------------------------


def run_agent_for_record(rec: Dict[str, Any]) -> str:
    rec = normalize_record(rec)
    # Build a concise user message that provides raw record JSON
    user_msg = (
        "Process this insurance claim record end-to-end using the mandatory tool sequence. "
        "Return only the final Decision and Reason at the end.\n\n"
        f"RAW_RECORD_JSON:\n{json.dumps(rec, ensure_ascii=False)}"
    )
    out = agent.invoke({"messages": [("system", SYSTEM_PROMPT), ("user", user_msg)]})
    final_msg = out["messages"][-1].content
    return final_msg.strip()


# -------------------------
# 6) Validation loop (optional)
# -------------------------


def run_validation():
    if not os.path.exists(VAL_RECORDS):
        print("validation_records.json not found; skip validation loop.")
        return []
    val_data = load_json(VAL_RECORDS)
    results = []
    for rec in val_data:
        pid = rec.get("patient_id", f"VAL_{uuid.uuid4().hex[:8]}")
        result = run_agent_for_record(rec)
        results.append({"patient_id": pid, "generated_response": result})
    df = pd.DataFrame(results)
    df.to_csv("validation_run.csv", index=False)
    return results


# -------------------------
# 7) Test loop -> submission.csv
# -------------------------


def run_test_and_write_submission():
    if not os.path.exists(TEST_RECORDS):
        print("test_records.json not found; cannot generate submission.csv")
        return
    test_data = load_json(TEST_RECORDS)
    rows = []
    for rec in test_data:
        pid = rec.get("patient_id", f"TEST_{uuid.uuid4().hex[:8]}")
        result = run_agent_for_record(rec)
        rows.append({"patient_id": pid, "generated_response": result})
    df = pd.DataFrame(rows)
    df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")


if __name__ == "__main__":
    # Optional: quick smoke test with a minimal inline record (remove in production)
    sample_record = {
        "patient_id": "P002",
        "name": "Robert Jones",
        "date_of_birth": "1982-03-15",
        "gender": "Female",
        "insurance_policy_id": "POL1007",
        "diagnosis_codes": ["N39.0"],
        "procedure_codes": ["85025"],
        "date_of_service": "2025-05-03",
        "billed_amount": 3500.0,
        "preauthorization_required": True,
        "preauthorization_obtained": True,
    }
    print(run_agent_for_record(sample_record))

    # Then run full test set
    # run_test_and_write_submission()
