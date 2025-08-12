from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from util.langgraph_util import display
from langgraph.checkpoint.memory import MemorySaver


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
model = ChatOpenAI().bind_tools(tools)


# TODO: Define functions for the workflow
def call_model(state: MessagesState):
    return None


# TODO: Define Conditional Routing
def should_continue(state: MessagesState):
    return None


# TODO: Define the workflow

workflow = StateGraph(MessagesState)


graph= workflow.compile()

display(graph)
# First invoke - Get one restaurant recommendation
response = graph.invoke(
    {"messages": [HumanMessage(content="Can you recommend just one top restaurant in Munich? "
                                       "The response should contain just the restaurant name")]})

# TODO: Extract the recommended restaurant
