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
insurance_file_json = os.path.join(current_dir, "data/insurance_policies.json")
reference_codes_file_json = os.path.join(current_dir, "data/reference_codes.json")

with open(insurance_file_json, "r") as file:
    insurance_data = json.load(file)

with open(reference_codes_file_json, "r") as file:
    reference_codes_data = json.load(file)


## 2. Data Chunking - Optimized for Insurance Policies
def create_policy_chunks(insurance_data):
    """
    Create optimized chunks for insurance policy data.
    Each policy becomes a separate chunk to maintain context integrity.
    """
    chunks = []

    for policy in insurance_data:
        # Create a comprehensive text representation of each policy
        policy_text = f"""Policy ID: {policy['policy_id']}
    Plan Name: {policy['plan_name']}

    Covered Procedures:"""

        for proc in policy["covered_procedures"]:
            procedure_desc = reference_codes_data["CPT"].get(
                proc["procedure_code"], "Description not found"
            )
            diagnoses_desc = [
                f"{diag} ({reference_codes_data['ICD10'].get(diag, 'Description not found')})"
                for diag in proc["covered_diagnoses"]
            ]
            procedure_text = f"""
    - Procedure Code: {proc['procedure_code']}
    - Description: {procedure_desc}
    - Covered Diagnoses: {', '.join(diagnoses_desc)}
    - Age Range: {proc['age_range'][0]}-{proc['age_range'][1]} years
    - Gender: {proc['gender']}
    - Requires Preauthorization: {proc['requires_preauthorization']}
    - Notes: {proc['notes']}"""
            policy_text += procedure_text

        chunks.append(
            {
                "text": policy_text.strip(),
                "metadata": {
                    "policy_id": policy["policy_id"],
                    "plan_name": policy["plan_name"],
                    "covered_procedures": ",".join(
                        [
                            f"procedure_code: {proc['procedure_code']},"
                            f"description: {reference_codes_data['CPT'].get(proc['procedure_code'], 'Description not found')},"
                            f"age_range: {proc['age_range'][0]}-{proc['age_range'][1]},"
                            f"gender: {proc['gender']},"
                            f"requires_preauthorization: {proc['requires_preauthorization']},"
                            f"notes: {proc['notes']},"
                            f"covered_diagnoses: {', '.join([f'{diag} ({reference_codes_data['ICD10'].get(diag, 'Description not found')})' for diag in proc['covered_diagnoses']])}"
                            for proc in policy["covered_procedures"]
                        ]
                    ),
                    "num_procedures": len(policy["covered_procedures"]),
                    "source": "insurance_policies.json",
                },
            }
        )

    return chunks


# Create chunks using custom function (recommended)
chunks = create_policy_chunks(insurance_data)


## 3. Vector Store Creation
# Extract texts and metadata for vector store
texts = [chunk["text"] for chunk in chunks]
metadatas = [chunk["metadata"] for chunk in chunks]


insurance_db = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embeddings,
    collection_name="insurance-policies",
)

query = "tell me about procedure code 36415"
result = insurance_db.similarity_search(query)
print(result)
