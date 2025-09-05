import os
import json
import csv
import re
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple

load_dotenv()

# Set your Azure OpenAI credentials
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]


# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
    temperature=0,
    request_timeout=60,
    max_retries=3,
)


##### utility functions #####
def read_json_file(file_path: str) -> dict:
    """
    Reads a JSON file from the 'data' directory relative to the current script location.
    - Uses robust error handling for missing files and invalid JSON.
    - Returns parsed JSON data as a Python dictionary.
    """
    try:
        # Get the absolute path to the 'data' directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, "data", file_path)

        # Check if the file exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        # Open and load the JSON file
        with open(full_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if not data:
                raise ValueError(f"Empty or invalid JSON in {file_path}")
        return data
    except json.JSONDecodeError as e:
        # Handle invalid JSON format
        raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")
    except Exception as e:
        # Propagate other exceptions
        raise


def calculate_age(date_of_birth: str, date_of_service: str) -> int:
    """
    Calculate the patient's age at the time of service.

    Args:
        date_of_birth (str): Patient's date of birth in 'YYYY-MM-DD' format.
        date_of_service (str): Date of service in 'YYYY-MM-DD' format.

    Returns:
        int: Age in years at the date of service.

    Raises:
        ValueError: If either date is not in the correct format.
    """
    try:
        # Parse the input date strings to datetime objects
        dob = datetime.strptime(date_of_birth, "%Y-%m-%d")
        dos = datetime.strptime(date_of_service, "%Y-%m-%d")
        # Calculate age, adjusting if birthday hasn't occurred yet in the service year
        age = dos.year - dob.year - ((dos.month, dos.day) < (dob.month, dob.day))
        return age
    except ValueError as e:
        # Raise a clear error if date parsing fails
        raise ValueError(f"Invalid date format: {e}")


reference_codes_data = read_json_file("reference_codes.json")
insurance_policies_data = read_json_file("insurance_policies.json")

############################


def create_policy_summary(policy_id):
    """
    Create a comprehensive policy summary with clear segregation of multiple covered procedures.
    Enhanced to prevent Azure OpenAI age range confusion.
    """
    policy = None
    for p in insurance_policies_data:
        if p["policy_id"] == policy_id:
            policy = p
            break

    if not policy:
        return f"Policy Details:\n- Policy ID: {policy_id}\n- Plan Name: Not Found\n\nCovered Procedures:\n- None"

    # Create policy header
    policy_summary = f"""Policy Details:
- Policy ID: {policy['policy_id']}
- Plan Name: {policy.get('plan_name', 'Unknown')}

Covered Procedures:"""
    covered_procedures = policy.get("covered_procedures", [])
    if not covered_procedures:
        policy_summary += "\n- No procedures covered under this policy"
        return policy_summary

    # Process each procedure with clear numbering and segregation
    for i, proc in enumerate(covered_procedures, 1):
        procedure_desc = reference_codes_data["CPT"].get(
            proc["procedure_code"], "Description not found"
        )

        # Format diagnoses with ICD-10 codes and descriptions
        diagnoses_desc = []
        for diag in proc.get("covered_diagnoses", []):
            icd_desc = reference_codes_data["ICD10"].get(diag, "Description not found")
            diagnoses_desc.append(f"{diag} ({icd_desc})")

        # FIXED: Format age range with explicit boundary explanation
        age_range = proc.get("age_range", [])
        if len(age_range) == 2:
            age_text = f"Ages {age_range[0]} to {age_range[1]-1} (inclusive) - Range: [{age_range[0]}, {age_range[1]})"
        else:
            age_text = "Not specified"

        # Create clearly segregated procedure section
        procedure_text = f"""

Procedure {i}:
- Procedure Code and Description:
  - code: {proc['procedure_code']}
  - description: {procedure_desc}
- Covered Diagnoses and Descriptions:
  - {', '.join(diagnoses_desc) if diagnoses_desc else 'None specified'}
- Gender Restriction:{proc.get('gender', 'Any')}
- Age Eligibility:{age_text}
- Preauthorization Requirement: {bool(proc.get('requires_preauthorization', False))}
- Notes on Coverage:{proc.get('notes', 'None')}"""

        policy_summary += procedure_text

    return policy_summary


def create_patient_record_summary(record_string):
    """
    Create optimized chunks for patient record data following the 7-section requirement.
    Each patient record becomes a separate chunk to maintain context integrity.
    """

    # Format diagnoses with ICD-10 codes and descriptions
    diagnoses_text = []
    for diag in record_string.get("diagnosis_codes", []):
        description = reference_codes_data["ICD10"].get(diag, "Description not found")
        diagnoses_text.append(f"code: {diag}, description: {description}")
    diagnoses_formatted = "\n  ".join(diagnoses_text) if diagnoses_text else "None"

    # Format procedures with CPT codes and descriptions
    procedures_text = []
    for proc in record_string.get("procedure_codes", []):
        description = reference_codes_data["CPT"].get(proc, "Description not found")
        procedures_text.append(f"code: {proc}, description : {description}")
    procedures_formatted = "\n  ".join(procedures_text) if procedures_text else "None"

    patient_age = calculate_age(
        record_string.get("date_of_birth", "Unknown"),
        record_string.get("date_of_service", "Unknown"),
    )
    # Create a comprehensive text representation following the 7-section requirement
    record_summary = f"""Patient Demographics:
  - Name: {record_string.get('name', 'Unknown')}
  - Gender: {record_string.get('gender', 'Unknown')}
  - Age: {patient_age} years

Insurance Policy ID: {record_string.get('insurance_policy_id', 'Unknown')}

Diagnoses and Descriptions:
  {diagnoses_formatted}

Procedures and Descriptions:
  {procedures_formatted}

Preauthorization Status:
  - Required:  {record_string.get('preauthorization_required', False)}
  - Obtained: {record_string.get('preauthorization_obtained', False)}

Billed Amount (in USD): ${record_string.get('billed_amount', 0):,.2f}

Date of Service: {record_string.get('date_of_service', 'Unknown')}"""

    return record_summary


######


#### Tools ######


@tool
def summarize_patient_record(record_string: str) -> str:
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
    try:

        if isinstance(record_string, str):
            record_string = json.loads(record_string)
        record_summary = create_patient_record_summary(record_string)
        return record_summary
    except FileNotFoundError as fnf_error:
        return f"Patient records file not found: {str(fnf_error)}"
    except json.JSONDecodeError as json_error:
        return f"Error decoding patient records JSON: {str(json_error)}"
    except Exception as e:
        return f"An unexpected error occurred while summarizing the patient record: {str(e)}"


@tool
def summarize_policy_guideline(policy_id: str) -> str:
    """
    Extract a structured summary of the insurance policy corresponding to the given policy_id.

    Input:
    - policy_id (str): Policy identifier (e.g., "POL1002")

    Output:
    - Structured policy summary with clear segregation of multiple covered procedures
    - Policy Details section with policy ID and plan name
    - Covered Procedures section with each procedure listed separately

    For each covered procedure, the following sub-sections are included:
    - Procedure Code and Description (using CPT code mappings from reference data)
    - Covered Diagnoses and Descriptions (using ICD-10 code mappings from reference data)
    - Gender Restriction (specifies allowed gender or "Any")
    - Age Range (format: [lower_bound, upper_bound) - inclusive lower, exclusive upper)
    - Preauthorization Requirement (boolean indicating if preauth is required)
    - Notes on Coverage (additional policy-specific notes or restrictions)

    The function handles policies with multiple procedures by clearly segregating each procedure's
    coverage criteria, making it easy to evaluate claims against specific procedure requirements.
    """
    try:
        policy_summary = create_policy_summary(policy_id)
        if not policy_summary:
            return f"No relevant policy guideline found for policy ID: {policy_id}"
        return policy_summary
    except FileNotFoundError:
        return "Insurance policies file not found."
    except Exception as e:
        return f"An error occurred while summarizing the policy guideline: {str(e)}"


@tool
def check_claim_coverage(record_summary: str, policy_summary: str) -> str:
    """
    Azure OpenAI insurance claim coverage evaluation with structured output format.

    Args:
        record_summary: Structured patient record from summarize_patient_record
        policy_summary: Structured policy details from summarize_policy_guideline

    Returns:
        Structured coverage decision with Decision and Reason format
    """
    try:
        # Updated prompt for structured Azure OpenAI output
        prompt = PromptTemplate(
            template="""You are an insurance claims analyst using Azure OpenAI. Evaluate this claim and provide a structured response.

=== PATIENT RECORD ===
{record_summary}

=== POLICY DETAILS ===
{policy_summary}

EVALUATION CRITERIA (ALL must be met for APPROVAL):
1. Diagnosis codes match policy-covered diagnoses for the procedure
2. Procedure code explicitly listed in policy
3. Patient age within policy range [lower, upper) - inclusive lower, exclusive upper
4. Patient gender matches policy requirement
5. If preauthorization required, it must be obtained

OUTPUT FORMAT (exactly as shown):


OUTPUT FORMAT (exactly as shown with three sections):

**Coverage Review:**
Step-by-step analysis for the claimed procedure:
- Procedure Verification: [Check if CPT code [code] for [procedure description] is covered under the policy]
- Diagnosis Matching: [Verify if patient's diagnosis [diagnosis code] - [diagnosis description] matches policy-covered diagnoses]
- Age Eligibility Check: [Verify if patient age [age] falls within policy age range [range]]
- Gender Requirement Check: [Verify if patient gender [gender] meets policy gender restriction]
- Preauthorization Verification: [Check preauthorization requirement and status]

**Summary of Findings:**
Coverage requirements assessment:
- Procedure Coverage: [MET/NOT MET] - [brief explanation]
- Diagnosis Matching: [MET/NOT MET] - [brief explanation]
- Age Eligibility: [MET/NOT MET] - [brief explanation]
- Gender Requirements: [MET/NOT MET] - [brief explanation]
- Preauthorization: [MET/NOT MET/NOT REQUIRED] - [brief explanation]

**Final Decision:**
- Decision: APPROVE
- Reason: The claim for [full procedure description] (CPT code [code]) has been approved. This procedure is covered under the policy for [treating/the diagnosis of] [diagnosis description] ([diagnosis code]), which [matches the patient's diagnosis/applies to the patient's diagnosis]. The patient, a [age]-year-old [gender], meets all eligibility criteria, and [preauthorization status - 'no preauthorization is required' or 'preauthorization was obtained' or 'preauthorization isn't required, but it was obtained'].

OR

- Decision: ROUTE FOR REVIEW  
- Reason: The claim for [full procedure description] (CPT code [code]) cannot be automatically approved[, as/due to] [specific reason]. [Additional context about what criteria are/aren't met]. [End with routing statement: 'Therefore, the claim needs to be routed for further manual review.' or 'As a result, the claim needs to be routed for further manual review.']

EXAMPLE OUTPUT:

**Coverage Review:**
Step-by-step analysis for the claimed procedure:
- Procedure Verification: CPT code 36415 for collecting venous blood by venipuncture is explicitly listed as a covered procedure under this policy
- Diagnosis Matching: Patient's diagnosis N39.0 (urinary tract infection) matches the policy-covered diagnosis for this procedure
- Age Eligibility Check: Patient age 16 falls within the policy age range [0, 100) for this procedure
- Gender Requirement Check: Patient gender meets the policy requirement
- Preauthorization Verification: No preauthorization is required for this procedure

**Summary of Findings:**
Coverage requirements assessment:
- Procedure Coverage: MET - CPT 36415 is explicitly covered under the policy
- Diagnosis Matching: MET - N39.0 matches policy-covered diagnoses
- Age Eligibility: MET - Age 16 is within allowed range [0, 100)
- Gender Requirements: MET - No gender restrictions apply
- Preauthorization: NOT REQUIRED - This procedure does not require preauthorization

**Final Decision:**
- Decision: APPROVE
- Reason: The claim for collecting venous blood by venipuncture (CPT code 36415) has been approved. This procedure is covered under the policy for treating urinary tract infection (N39.0), which matches the patient's diagnosis. The patient, a 16-year-old female, meets all eligibility criteria, and no preauthorization is required.

- Decision: ROUTE FOR REVIEW
- Reason: The claim for the complete blood count (CPT code 85025) cannot be automatically approved, as the patient's age of 46 exceeds the policy's allowed age range of 18 to 45. Although the diagnosis of major depressive disorder (F32.9) is covered, the age requirement is not met. Therefore, the claim needs to be routed for further manual review.

Generate structured response:""",
            input_variables=["record_summary", "policy_summary"],
        )

        # Azure OpenAI processing
        chain = prompt | llm
        response = chain.invoke(
            {"record_summary": record_summary, "policy_summary": policy_summary}
        )

        return response.content.strip()

    except Exception as e:
        return f"Decision: ROUTE FOR REVIEW\nReason: The claim cannot be processed due to a system error: {str(e)}. The claim needs to be routed for manual review."


### end of tools ####


## tool list
tools = [summarize_patient_record, summarize_policy_guideline, check_claim_coverage]

# -------------------------
# 3) Agent system prompt
# -------------------------

AGENT_PROMPT_TXT = """You are a ReAct-style, policy-compliant Insurance Claim Approval Agent using Azure OpenAI.

TOOLS YOU MAY USE (and only these):
1) summarize_patient_record(record_str) - Extract structured patient data
2) summarize_policy_guideline(policy_id) - Get policy coverage details  
3) check_claim_coverage(record_summary, policy_summary) - Make coverage decision

MANDATORY WORKFLOW (strict order):
Step A: Call summarize_patient_record(record_str) on the raw patient JSON/text
Step B: Extract policy_id from Step A output, then call summarize_policy_guideline(policy_id)
Step C: Call check_claim_coverage(record_summary, policy_summary) using outputs from Steps A and B


CRITICAL OUTPUT REQUIREMENTS:
- The check_claim_coverage tool returns a structured three-section analysis
- You MUST extract ONLY the Final Decision section and reformat it as follows:

REQUIRED FINAL OUTPUT FORMAT (exactly as shown):
- Decision: [APPROVE or ROUTE FOR REVIEW]
- Reason: [Single comprehensive sentence explaining the decision based on specific policy criteria, procedure coverage, patient eligibility, diagnosis code and preauthorization status]

DECISION EXTRACTION RULES:
- Look for "**Final Decision:**" section in the tool output
- Extract the Decision line (APPROVE or ROUTE FOR REVIEW)
- Extract the Reason line and ensure it references:
  * Specific procedure code and description
  * Relevant policy coverage criteria
  * Patient eligibility factors (age, gender, diagnosis)
  * Preauthorization status if applicable
  * Specific policy rule that determined the outcome

EXAMPLE FINAL OUTPUT:
Decision: APPROVE
Reason: The claim for collecting venous blood by venipuncture (CPT code 36415) has been approved. This procedure is covered under the policy for treating urinary tract infection (N39.0), which matches the patient’s diagnosis. The patient, a 16-year-old female, meets all eligibility criteria, and no preauthorization is required.

OR 

Decision: ROUTE FOR REVIEW
Reason: The claim for the complete blood count (CPT code 85025) cannot be automatically approved, as the patient's age of 46 exceeds the policy's allowed age range of 18 to 45. Although the diagnosis of major depressive disorder (F32.9) is covered, the age requirement is not met. Therefore, the claim needs to be routed for further manual review.



Process the claim following this exact sequence and output format.

"""


health_insurance_claim_approval_agent = create_react_agent(
    model=llm,
    tools=tools,
)


# Utility function to call the agent and stream its step-by-step reasoning
def call_agent(agent, query, verbose=False, max_retries=3):
    """Enhanced agent call with retry logic and better error handling"""
    import time

    for attempt in range(max_retries):
        try:
            # Include system prompt in messages
            messages = [
                ("system", AGENT_PROMPT_TXT),
                ("human", f"Process this insurance claim record end-to-end: {query}"),
            ]

            # Stream the agent's execution
            for event in agent.stream(
                {"messages": messages},
                stream_mode="values",
            ):
                if verbose:
                    event["messages"][-1].pretty_print()

            final_response = event["messages"][-1].content

            # Validate response format
            if not final_response or len(final_response.strip()) < 10:
                raise ValueError("Invalid or empty response from agent")

            return final_response.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2**attempt)  # Exponential backoff
                continue
            else:
                return f"The claim cannot be processed due to a system error after {max_retries} attempts: {str(e)}. The claim needs to be routed for manual review."


def run_validation():
    results = []
    # validation_record = read_json_file("test_records.json")
    validation_record = read_json_file("test_records.json")
    for record in validation_record:
        query = json.dumps(record)
        print(f"\nProcessing claim for Patient ID: {record['patient_id']}")
        result = call_agent(health_insurance_claim_approval_agent, query, verbose=True)
        # cleaned_result = result.replace("\n", " ").replace("\r", " ").strip()
        results.append((record["patient_id"], result))
    # Ensure results is a list of tuples (patient_id, response)
    df = pd.DataFrame(results, columns=["patient_id", "generated_response"])
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "submission.csv"
    )
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",  # Ensure consistent line endings
    )
    print(f"Results written to {output_path}")
    # print(f"Query: {query}\nResponse: {result}")


def extract_decision_and_reason_from_agent_response(
    agent_response: str,
) -> Tuple[str, str]:
    """
    Extract Decision and Reason from Azure OpenAI React Agent response.

    Args:
        agent_response: Agent's complete response containing Decision and Reason

    Returns:
        tuple: (decision, reason) cleaned and formatted for evaluation
    """
    try:
        # Look for Decision pattern in agent response
        decision_patterns = [
            r"Decision:\s*(APPROVE|ROUTE FOR REVIEW)",
            r"-\s*Decision:\s*(APPROVE|ROUTE FOR REVIEW)",
            r"Final\s*Decision.*?Decision:\s*(APPROVE|ROUTE FOR REVIEW)",
        ]

        decision = "ROUTE FOR REVIEW"  # Default fallback
        for pattern in decision_patterns:
            match = re.search(pattern, agent_response, re.IGNORECASE | re.DOTALL)
            if match:
                decision = match.group(1).upper()
                break

        # Look for Reason pattern in agent response
        reason_patterns = [
            r"Reason:\s*(.+?)(?:\n\n|\n(?=[A-Z])|$)",
            r"-\s*Reason:\s*(.+?)(?:\n\n|\n(?=[A-Z])|$)",
            r"Final\s*Decision.*?Reason:\s*(.+?)(?:\n\n|\n(?=[A-Z])|$)",
        ]

        reason = "No reason extracted from agent response"  # Default fallback
        for pattern in reason_patterns:
            match = re.search(pattern, agent_response, re.IGNORECASE | re.DOTALL)
            if match:
                reason = match.group(1).strip()
                # Clean up the reason text
                reason = " ".join(reason.split())  # Normalize whitespace
                reason = reason.rstrip(".") + "."  # Ensure single period at end
                break

        return decision, reason

    except Exception as e:
        return "ROUTE FOR REVIEW", f"Error parsing agent response: {str(e)}"


def load_reference_responses(
    reference_file: str = "validation_reference_results.csv",
) -> Dict[str, str]:
    """
    Load reference responses for LLM-as-a-Judge evaluation.

    Args:
        reference_file: CSV file with reference responses

    Returns:
        Dict mapping patient_id to reference response
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ref_path = os.path.join(current_dir, "data", reference_file)

        if not os.path.exists(ref_path):
            print(f"Reference file not found: {ref_path}")
            return {}

        ref_df = pd.read_csv(ref_path)

        # Assuming columns are 'patient_id' and 'reference_response'
        # Adjust column names based on your actual CSV structure
        if "patient_id" in ref_df.columns:
            if "reference_response" in ref_df.columns:
                return dict(zip(ref_df["patient_id"], ref_df["reference_response"]))
            elif "response" in ref_df.columns:
                return dict(zip(ref_df["patient_id"], ref_df["response"]))
            else:
                # Use the second column as reference response
                return dict(zip(ref_df.iloc[:, 0], ref_df.iloc[:, 1]))
        else:
            print("Could not find patient_id column in reference file")
            return {}

    except Exception as e:
        print(f"Error loading reference responses: {e}")
        return {}


def llm_judge_single_evaluation(
    patient_id: str, agent_decision: str, agent_reason: str, reference_response: str
) -> Dict:
    """
    Evaluate a single React Agent response using Azure OpenAI LLM-as-a-Judge.

    Args:
        patient_id: Patient identifier
        agent_decision: React Agent's decision (APPROVE/ROUTE FOR REVIEW)
        agent_reason: React Agent's reasoning
        reference_response: Human reference response

    Returns:
        Dict containing evaluation scores and feedback
    """

    judge_prompt = f"""You are an Azure OpenAI expert insurance claims evaluator acting as a judge.

EVALUATION TASK: Compare the React Agent's response against the reference response for insurance claim processing.

PATIENT ID: {patient_id}

REACT AGENT RESPONSE:
Decision: {agent_decision}
Reason: {agent_reason}

REFERENCE RESPONSE:
{reference_response}

EVALUATION CRITERIA:
1. Decision Accuracy (50%): Does the agent's decision (APPROVE/ROUTE FOR REVIEW) match the reference?
2. Reasoning Quality (30%): Does the agent's reason cover the same key points as reference?
3. Completeness (20%): Does the agent mention procedure codes, diagnoses, age, policy criteria?

SCORING SCALE (0-10):
- 9-10: Excellent - Decision matches, reasoning comprehensive and accurate
- 7-8: Good - Decision matches, reasoning mostly complete with minor gaps
- 5-6: Satisfactory - Decision matches but reasoning lacks some key elements
- 3-4: Poor - Decision matches but reasoning significantly incomplete or incorrect
- 1-2: Very Poor - Decision mismatches or reasoning fundamentally flawed
- 0: Unacceptable - Completely incorrect or incomprehensible

OUTPUT FORMAT:
OVERALL_SCORE: [0-10]
DECISION_ACCURACY: [MATCH/MISMATCH] - [explanation]
REASONING_QUALITY: [EXCELLENT/GOOD/FAIR/POOR] - [explanation]
COMPLETENESS: [COMPLETE/PARTIAL/INCOMPLETE] - [explanation]
IMPROVEMENT_SUGGESTIONS: [specific recommendations for React Agent]

Provide your Azure OpenAI evaluation:"""

    try:
        # Azure OpenAI judge evaluation
        evaluation_response = llm.invoke(judge_prompt)
        evaluation_text = evaluation_response.content

        # Parse evaluation results
        score_match = re.search(r"OVERALL_SCORE:\s*(\d+)", evaluation_text)
        overall_score = int(score_match.group(1)) if score_match else 0

        decision_match = re.search(
            r"DECISION_ACCURACY:\s*(\w+)\s*-\s*(.+?)(?=REASONING_QUALITY:|$)",
            evaluation_text,
            re.DOTALL,
        )
        decision_accuracy = decision_match.group(1) if decision_match else "UNKNOWN"
        decision_explanation = (
            decision_match.group(2).strip() if decision_match else "No explanation"
        )

        reasoning_match = re.search(
            r"REASONING_QUALITY:\s*(\w+)\s*-\s*(.+?)(?=COMPLETENESS:|$)",
            evaluation_text,
            re.DOTALL,
        )
        reasoning_quality = reasoning_match.group(1) if reasoning_match else "UNKNOWN"
        reasoning_explanation = (
            reasoning_match.group(2).strip() if reasoning_match else "No explanation"
        )

        completeness_match = re.search(
            r"COMPLETENESS:\s*(\w+)\s*-\s*(.+?)(?=IMPROVEMENT_SUGGESTIONS:|$)",
            evaluation_text,
            re.DOTALL,
        )
        completeness = completeness_match.group(1) if completeness_match else "UNKNOWN"
        completeness_explanation = (
            completeness_match.group(2).strip()
            if completeness_match
            else "No explanation"
        )

        suggestions_match = re.search(
            r"IMPROVEMENT_SUGGESTIONS:\s*(.+)", evaluation_text, re.DOTALL
        )
        suggestions = (
            suggestions_match.group(1).strip()
            if suggestions_match
            else "No suggestions provided"
        )

        return {
            "patient_id": patient_id,
            "overall_score": overall_score,
            "decision_accuracy": decision_accuracy,
            "decision_explanation": decision_explanation,
            "reasoning_quality": reasoning_quality,
            "reasoning_explanation": reasoning_explanation,
            "completeness": completeness,
            "completeness_explanation": completeness_explanation,
            "improvement_suggestions": suggestions,
            "raw_evaluation": evaluation_text,
        }

    except Exception as e:
        return {
            "patient_id": patient_id,
            "overall_score": 0,
            "decision_accuracy": "ERROR",
            "decision_explanation": f"Azure evaluation error: {str(e)}",
            "reasoning_quality": "ERROR",
            "reasoning_explanation": f"Azure evaluation error: {str(e)}",
            "completeness": "ERROR",
            "completeness_explanation": f"Azure evaluation error: {str(e)}",
            "improvement_suggestions": f"Fix Azure OpenAI evaluation error: {str(e)}",
            "raw_evaluation": "",
        }


def run_validation_with_llm_judge():
    """
    Enhanced Azure OpenAI React Agent validation with LLM-as-a-Judge evaluation.
    """
    print(
        "Starting Azure OpenAI React Agent validation with LLM-as-a-Judge evaluation..."
    )

    # Load reference responses
    reference_responses = load_reference_responses()
    if not reference_responses:
        print(
            "Warning: No reference responses loaded. LLM-as-a-Judge evaluation will be limited."
        )

    results = []
    judge_evaluations = []
    validation_record = read_json_file("validation_records.json")

    for record in validation_record:
        query = json.dumps(record)
        patient_id = record["patient_id"]

        print(f"\nProcessing claim for Patient ID: {patient_id}")

        # Get React Agent response
        full_agent_response = call_agent(
            health_insurance_claim_approval_agent, query, verbose=True
        )

        # Extract Decision and Reason
        agent_decision, agent_reason = extract_decision_and_reason_from_agent_response(
            full_agent_response
        )

        # Store for CSV output (Decision and Reason only)
        results.append(
            {
                "patient_id": patient_id,
                "Decision": agent_decision,
                "Reason": agent_reason,
            }
        )

        print(f"Extracted Decision: {agent_decision}")
        print(
            f"Extracted Reason: {agent_reason[:100]}..."
            if len(agent_reason) > 100
            else f"Extracted Reason: {agent_reason}"
        )

        # Run LLM-as-a-Judge evaluation if reference available
        if patient_id in reference_responses:
            print(f"Running LLM-as-a-Judge evaluation for Patient {patient_id}...")

            reference_response = reference_responses[patient_id]
            evaluation = llm_judge_single_evaluation(
                patient_id, agent_decision, agent_reason, reference_response
            )
            judge_evaluations.append(evaluation)

            print(f"LLM Judge Score: {evaluation['overall_score']}/10")
            print(f"Decision Accuracy: {evaluation['decision_accuracy']}")
        else:
            print(f"No reference response found for Patient {patient_id}")

    # Save CSV with Decision and Reason columns only
    csv_df = pd.DataFrame(results)
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "submission.csv"
    )
    csv_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    print(f"CSV results written to: {output_path}")

    # Display LLM-as-a-Judge evaluation results in console
    if judge_evaluations:
        judge_df = pd.DataFrame(judge_evaluations)

        # Print summary statistics
        avg_score = judge_df["overall_score"].mean()
        decision_accuracy_rate = (
            (judge_df["decision_accuracy"] == "MATCH").sum() / len(judge_df) * 100
        )

        print(f"\n{'='*80}")
        print("AZURE OPENAI LLM-AS-A-JUDGE EVALUATION RESULTS")
        print(f"{'='*80}")

        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Average Overall Score: {avg_score:.2f}/10")
        print(f"  Decision Accuracy Rate: {decision_accuracy_rate:.1f}%")
        print(f"  Total Cases Evaluated: {len(judge_df)}")

        # Display detailed evaluation DataFrame in console
        print(f"\nDETAILED EVALUATION RESULTS:")
        print("-" * 80)

        # Create a summary DataFrame for console display
        display_df = judge_df[
            [
                "patient_id",
                "overall_score",
                "decision_accuracy",
                "reasoning_quality",
                "completeness",
            ]
        ].copy()

        # Set display options for better formatting
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 15)

        print(display_df.to_string(index=False))

        # Quality distribution
        excellent = (judge_df["overall_score"] >= 9).sum()
        good = (
            (judge_df["overall_score"] >= 7) & (judge_df["overall_score"] < 9)
        ).sum()
        satisfactory = (
            (judge_df["overall_score"] >= 5) & (judge_df["overall_score"] < 7)
        ).sum()
        poor = (judge_df["overall_score"] < 5).sum()
        total_cases = len(judge_df)

        print(f"\nQUALITY DISTRIBUTION:")
        print(
            f"  Excellent (9-10): {excellent} cases ({excellent/total_cases*100:.1f}%)"
        )
        print(f"  Good (7-8): {good} cases ({good/total_cases*100:.1f}%)")
        print(
            f"  Satisfactory (5-6): {satisfactory} cases ({satisfactory/total_cases*100:.1f}%)"
        )
        print(f"  Poor (<5): {poor} cases ({poor/total_cases*100:.1f}%)")

        print(f"\n{'='*80}")

        return judge_df
    else:
        print("No LLM-as-a-Judge evaluations performed (no reference responses found)")
        return pd.DataFrame()


# Replace the original run_validation() call
# run_validation()
judge_results = run_validation_with_llm_judge()
