from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

# from util.langgraph_util import display
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

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
)


@tool
def get_restaurant_recommendations(location: str):
    """Provides a single top restaurant recommendation for a given location."""
    recommendations = {
        "munich": ["Hofbräuhaus", "Augustiner-Keller", "Tantris"],
        "new york": ["Le Bernardin", "Eleven Madison Park", "Joe's Pizza"],
        "paris": ["Le Meurice", "L'Ambroisie", "Bistrot Paul Bert"],
    }
    return recommendations.get(location.lower(), ["No recommendations available."])


@tool
def book_table(restaurant: str, time: str):
    """Books a table at a specified restaurant and time."""
    return f"Table booked at {restaurant} for {time}."


# Bind the tool to the model
tools = [get_restaurant_recommendations, book_table]
model = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# TODO: Define functions for the workflow
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


# TODO: Define Conditional Routing
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_mesage = messages[-1]
    if last_mesage.tool_calls:
        return "tools"
    return END


# TODO: Define the workflow

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)


workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()

graph = workflow.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "1",
    }
}  # Define configuration for memory

# display(graph)
# First invoke - Get one restaurant recommendation
response = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Can you recommend jst one top restaurant in Munich? "
                "The response should contain just the restaurant name",
            )
        ]
    },
    config=config,
)

# TODO: Extract the recommended restaurant
recommended_resturant = response["messages"][-1].content
print(recommended_resturant)


# Second invoke - Book Table
response = graph.invoke(
    {"messages": [HumanMessage(content="Book a table at this restaurant for 7 PM")]},
    config=config,
)

# TODO: Extract the recommended restaurant
book_table = response["messages"][-1].content
print(book_table)
