import os
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Set your Azure OpenAI credentials
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
embeddings_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_FOR_EMBEDDING"]

# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embeddings_deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


## 1. Data loading
current_dir = os.path.dirname(os.path.abspath(__file__))
insurance_file_json = os.path.join(current_dir, "data/validation_records.json")
reference_codes_file_json = os.path.join(current_dir, "data/reference_codes.json")

with open(insurance_file_json, "r") as file:
    patient_data = json.load(file)

with open(reference_codes_file_json, "r") as file:
    reference_codes_data = json.load(file)


with open(insurance_file_json, "r") as file:
    patient_data = json.load(file)


def calculate_age(date_of_birth, date_of_service):
    from datetime import datetime

    dob = datetime.strptime(date_of_birth, "%Y-%m-%d")
    dos = datetime.strptime(date_of_service, "%Y-%m-%d")
    age = dos.year - dob.year - ((dos.month, dos.day) < (dob.month, dob.day))
    return age


# print(calculate_age("1982-03-15", "2025-05-03"))


def create_patient_chunks(patient_data):
    """
    Create optimized chunks for patient record data following the 7-section requirement.
    Each patient record becomes a separate chunk to maintain context integrity.
    """
    chunks = []

    for record in patient_data:
        # Format diagnoses with ICD-10 codes and descriptions
        diagnoses_text = []
        for diag in record["diagnosis_codes"]:
            description = reference_codes_data["ICD10"].get(
                diag, "Description not found"
            )
            diagnoses_text.append(f"code: {diag}, description: {description}")
        diagnoses_formatted = "\n  ".join(diagnoses_text) if diagnoses_text else "None"

        # Format procedures with CPT codes and descriptions
        procedures_text = []
        for proc in record["procedure_codes"]:
            description = reference_codes_data["CPT"].get(proc, "Description not found")
            procedures_text.append(f"code: {proc}, description : {description}")
        procedures_formatted = (
            "\n  ".join(procedures_text) if procedures_text else "None"
        )

        # Create a comprehensive text representation following the 7-section requirement
        record_text = f"""Patient Demographics:
  Name: {record['name']}
  Gender: {record['gender']}
  Age: {calculate_age(record['date_of_birth'], record['date_of_service'])} years

Insurance Policy ID: {record['insurance_policy_id']}

Diagnoses and Descriptions:
  {diagnoses_formatted}

Procedures and Descriptions:
  {procedures_formatted}

Preauthorization Status:
  Required: {"Yes" if record['preauthorization_required'] else "No"}
  Obtained: {"Yes" if record['preauthorization_obtained'] else "No"}

Billed Amount (in USD): ${record['billed_amount']:,.2f}

Date of Service: {record['date_of_service']}"""

        chunks.append(
            {
                "text": record_text.strip(),
                "metadata": {
                    "patient_id": record["patient_id"],
                    "name": record["name"],
                    "gender": record["gender"],
                    "age": calculate_age(
                        record["date_of_birth"], record["date_of_service"]
                    ),
                    "insurance_policy_id": record["insurance_policy_id"],
                    "date_of_service": record["date_of_service"],
                    "diagnoses": ", ".join(
                        [
                            f"code: {diag}, description: {reference_codes_data['ICD10'].get(diag, '')}"
                            for diag in record["diagnosis_codes"]
                        ]
                    ),
                    "procedures": ", ".join(
                        [
                            f"code: {proc}, description: {reference_codes_data['CPT'].get(proc, '')}"
                            for proc in record["procedure_codes"]
                        ]
                    ),
                    "preauthorization_required": record["preauthorization_required"],
                    "preauthorization_obtained": record["preauthorization_obtained"],
                    "billed_amount": record["billed_amount"],
                    "provider_id": record["provider_id"],
                    "provider_specialty": record["provider_specialty"],
                    "location": record["location"],
                    "source": "validation_records.json",
                },
            }
        )

    return chunks


patient_data_chunks = create_patient_chunks(patient_data)

patients_db = Chroma.from_texts(
    texts=[chunk["text"] for chunk in patient_data_chunks],
    metadatas=[chunk["metadata"] for chunk in patient_data_chunks],
    embedding=embeddings,
    collection_name="patient-summary",
)


query = "tell me about patients in Cardiology"
result = patients_db.similarity_search(query)
print(result)
