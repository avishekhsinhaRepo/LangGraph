from typing import TypedDict
from psycopg import connect
from psycopg.rows import dict_row
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver


DB_CONNECTION_STRING = "postgresql://localhost:5432/langgraph_memory"


# Define state format
class ChatState(TypedDict):
    messages: list


# Define tool for restaurant recommendations
@tool
def get_restaurant_recommendations(location: str):
    """Provides a single top restaurant recommendation for a given location."""
    recommendations = {
        "munich": ["Hofbräuhaus", "Augustiner-Keller", "Tantris"],
        "new york": ["Le Bernardin", "Eleven Madison Park", "Joe's Pizza"],
        "paris": ["Le Meurice", "L'Ambroisie", "Bistrot Paul Bert"],
    }
    return recommendations.get(location.lower(), ["No recommendations available."])[0]  # Return only the top recommendation


# Define tool for table booking
@tool
def book_table(restaurant: str, time: str):
    """Books a table at a specified restaurant and time."""
    return f"Table booked at {restaurant} for {time}."


# Bind tools to the model
tools = [get_restaurant_recommendations, book_table]
model = ChatOpenAI().bind_tools(tools)
tool_node = ToolNode(tools)


# Define function to invoke the model
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": messages + [response]}  # Add response to the state


# Define conditional routing logic
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Establish a database connection
with connect(DB_CONNECTION_STRING, autocommit=True, prepare_threshold=0, row_factory=dict_row) as conn:
    # TODO: Initialize Postgres-backed memory
    

    # Create the workflow graph
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    # TODO: Compile the graph with PostgreSQL-backed memory
    graph = workflow.compile(checkpointer=None)

    # Define configuration for memory
    config = {"configurable": {"thread_id": "1"}}

    # First invoke - Get one restaurant recommendation
    response = graph.invoke(
        {"messages": [HumanMessage(content="Can you recommend just one top restaurant in Paris? "
                                           "The response should contain just the restaurant name")]},
        config
    )

    # Extract the recommended restaurant from the response
    recommended_restaurant = response["messages"][-1].content
    print("Recommended restaurant:", recommended_restaurant)


    # Second invoke - Book a table at the recommended restaurant
    response = graph.invoke(
        {"messages": [HumanMessage(content=f"Book a table at this restaurant for 7 PM")]},
        config
    )

    # Extract the final response
    final_response = response["messages"][-1].content
    print("Final response:", final_response)
