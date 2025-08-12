from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI


# Define state
class MarketResearchState(TypedDict):
    query: str
    trends: str
    competitors: str
    sentiment: str
    summary: str


llm = ChatOpenAI()


def fetch_trends(state: MarketResearchState):
    response = llm.invoke(f"What are the latest market trends for {state['query']}?")
    return {"trends": response.content}


def analyze_competitors(state: MarketResearchState):
    response = llm.invoke(f"List top competitors in {state['query']} market.")
    return {"competitors": response.content}


def extract_sentiment(state: MarketResearchState):
    response = llm.invoke(f"What do customers feel about products in {state['query']} category?")
    return {"sentiment": response.content}


def summarize(state: MarketResearchState):
    summary_prompt = f"""
    Product Research Summary:
    - Trends: {state.get('trends')}
    - Competitors: {state.get('competitors')}
    - Customer Sentiment: {state.get('sentiment')}
    Provide strategic insights for entering the {state['query']} market.
    """
    response = llm.invoke(summary_prompt)
    return {"summary": response.content}



graph_builder = StateGraph(MarketResearchState)

# Add nodes
graph_builder.add_node("fetch_trends", fetch_trends)
graph_builder.add_node("analyze_competitors", analyze_competitors)
graph_builder.add_node("extract_sentiment", extract_sentiment)
graph_builder.add_node("summarize", summarize)

# TODO: Add edges for parallel execution

# Compile graph
graph = graph_builder.compile()

# Run it
inputs = {"query": "Smart Water Bottle"}
result = graph.invoke(inputs)

# Output
print("\n=== Final Market Summary ===\n")
print(result["summary"])