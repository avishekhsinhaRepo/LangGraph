from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from operator import add
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage


# Define chatbot state with accumulated messages
class ChatBotState(TypedDict):
    messages: list[AnyMessage]
    discount: int


# Responses based on intent level
def connect_to_sales(state: ChatBotState):
    return {"messages": [AIMessage(content="Great! Let me connect you with our sales team right away. 🚀")],
            "discount": 10}


def sales_response(state: ChatBotState):
    return {"messages": [AIMessage(content="We have the best offer for you 🚀")],
            "discount": 20}


# Build chatbot conversation flow
graph_builder = StateGraph(ChatBotState)

# Add nodes

graph_builder.add_node("connect_to_sales", connect_to_sales)
graph_builder.add_node("sales_response", sales_response)

# Define conversation flow
graph_builder.add_edge(START, "connect_to_sales")
graph_builder.add_edge("connect_to_sales", "sales_response")
graph_builder.add_edge("sales_response", END)

# Compile chatbot
chatbot = graph_builder.compile()

# Simulate different conversations
test_inputs = "I want to buy your product."


messages = chatbot.invoke({"messages": [HumanMessage(content=test_inputs)]})

for message in messages["messages"]:
    print(f"🤖 **Bot:** {message.content}")


print("Final Discount: ",messages['discount'],'%')