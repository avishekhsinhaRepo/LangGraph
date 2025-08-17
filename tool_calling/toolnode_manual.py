from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage


@tool
def get_restaurant_recommendations(location: str):
    """Provides a list of top restaurant recommendations for a given location."""
    recommendations = {
        "munich": ["Hofbräuhaus", "Augustiner-Keller", "Tantris"],
        "new york": ["Le Bernardin", "Eleven Madison Park", "Joe's Pizza"],
        "paris": ["Le Meurice", "L'Ambroisie", "Bistrot Paul Bert"],
    }
    return recommendations.get(
        location.lower(), ["No recommendations available for this location."]
    )


tools = [get_restaurant_recommendations]
tool_node = ToolNode(tools)


# TODO: Create an AIMessage for the tool call
message_with_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_restaurant_recommendations",
            "args": {"location": "Munich"},
            "id": "call_mj9gU841Xr4IZRRb4Q3hLTI8",
            "type": "tool_call",
        }
    ],
)

# TODO: Invoke the ToolNode with the state and get the result
result = tool_node.invoke(
    {"messages": [message_with_tool_call]}
)  # Pass as a list of messages

# TODO: Output the result
print(result)
