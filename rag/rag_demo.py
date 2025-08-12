from typing import List, TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

# Current Affairs News Sources
news_urls = [
    "https://www.bbc.com/news",
    "https://www.cnn.com/world",
    "https://www.nytimes.com/section/world",
    "https://www.reuters.com/world/",
    "https://www.aljazeera.com/news/"
]

# Load Current Affairs Documents
docs = [WebBaseLoader(url).load() for url in news_urls]
docs_list = [item for sublist in docs for item in sublist]

# Split the articles for embeddings
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=20
)
doc_splits = text_splitter.split_documents(docs_list)

# Store and Retrieve Current Affairs with ChromaDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="current-affairs-news",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

# Prompt for Current Affairs News Summarization
prompt = ChatPromptTemplate.from_template(
    """
    You are a news analyst summarizing the latest current affairs.
    Use the retrieved articles to provide a concise summary.
    Highlight key global events and developments.

    Question: {question}
    News Articles: {context}
    Summary:
    """
)
model = ChatOpenAI()
current_affairs_chain = (
        prompt | model | StrOutputParser()
)


class CurrentAffairsGraphState(TypedDict):
    question: str
    retrieved_news: List[str]
    generation: str


# TODO: Use the retriever and retrieve the matching news
def retrieve_current_affairs(state):
    print("---RETRIEVE CURRENT AFFAIRS---")
    return None


# TODO: Summarize the news
# News Summary Generation Node
def generate_current_affairs_summary(state):
    print("---GENERATE CURRENT AFFAIRS SUMMARY---")
    return None

# Current Affairs News Workflow Definition
def create_current_affairs_workflow():
    workflow = StateGraph(CurrentAffairsGraphState)
    workflow.add_node("retrieve_current_affairs", retrieve_current_affairs)
    workflow.add_node("generate_current_affairs_summary", generate_current_affairs_summary)
    workflow.add_edge(START, "retrieve_current_affairs")
    workflow.add_edge("retrieve_current_affairs", "generate_current_affairs_summary")
    workflow.add_edge("generate_current_affairs_summary", END)
    return workflow.compile()


# Execute the Current Affairs News Workflow
current_affairs_graph = create_current_affairs_workflow()

inputs = {"question": "What are the top global headlines today?"}

response = current_affairs_graph.invoke(inputs)

print("\n--- CURRENT AFFAIRS SUMMARY ---")
# TODO: Print Response
