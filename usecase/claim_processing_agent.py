from typing import TypedDict, List
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import psycopg
import requests
from langgraph.types import interrupt
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


# ---------------------- Define State ----------------------
class ClaimState(TypedDict):
    patient_id: str
    treatment_code: str
    claim_details: str
    patient_data: dict
    insurance_data: dict
    policy_docs: List[str]
    ai_validation_feedback: str
    final_decision: str
    _next: str  # ✅ Added _next for decision-making


# ---------------------- Constants ----------------------
FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"

DB_CONFIG = {
    "dbname": "claims_db",
    "user": "postgres",
    "password": "test",
    "host": "localhost"
}

llm = ChatOpenAI()

# Load policy documents for RAG
loader = TextLoader("insurance_data.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)


# ---------------------- Step 1: Fetch Patient Data ----------------------
def fetch_patient_data(state: ClaimState):
    pass

# ---------------------- Step 2: Fetch Insurance Data ----------------------
def fetch_patient_insurance(state: ClaimState):
    pass


# ---------------------- Step 3: Retrieve Policy Documents ----------------------
def retrieve_policy_docs(state: ClaimState):
    pass


# ---------------------- Step 4: AI-Based Claim Validation ----------------------
def validate_claim(state: ClaimState):
    pass


# ---------------------- Step 5: Decision Node ----------------------
def claim_decision(state: ClaimState):
    pass


# ---------------------- Step 6: Store Decision in Database ----------------------
def store_claim(state: ClaimState):
    pass


# ---------------------- Step 7: Human Review ----------------------
def human_review(state: ClaimState):
    pass

# ---------------------- Build LangGraph Workflow ----------------------


def create_workflow():
    graph = StateGraph(ClaimState)
    graph.add_node("fetch_patient_data", fetch_patient_data)
    graph.add_node("fetch_patient_insurance", fetch_patient_insurance)
    graph.add_node("retrieve_policy_docs", retrieve_policy_docs)
    graph.add_node("validate_claim", validate_claim)
    graph.add_node("claim_decision", claim_decision)
    graph.add_node("store_claim", store_claim)
    graph.add_node("human_review", human_review)

    # ✅ Define workflow transitions
    graph.set_entry_point("fetch_patient_data")
    graph.add_edge("fetch_patient_data", "fetch_patient_insurance")
    graph.add_edge("fetch_patient_insurance", "retrieve_policy_docs")
    graph.add_edge("retrieve_policy_docs", "validate_claim")
    graph.add_edge("validate_claim", "claim_decision")
    graph.add_edge("human_review", "store_claim")

    # ✅ Decision-making edges
    graph.add_conditional_edges(
        "claim_decision",
        lambda state: state["_next"],
        {
            "store_claim": "store_claim",
            "human_review": "human_review"
        }
    )

    # ✅ Create an InMemoryCheckpointer
    checkpointer = MemorySaver()
    graph = graph.compile(checkpointer=checkpointer)
    return graph