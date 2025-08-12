from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from operator import add
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage


# Define chatbot state with accumulated orders
class OrderState(TypedDict):
    messages: list[AnyMessage]
    order_id: int


# Step 1: Take the food order
def take_order(state: OrderState):
    return {"messages": [AIMessage(content="Processing your order?")]}


# Step 2: Confirm the order
def confirm_order(state: OrderState):
    return {"messages": [AIMessage(content="Your order has been placed!")], "order_id": 1}


# Build chatbot conversation flow
graph_builder = StateGraph(OrderState)

# Add nodes
graph_builder.add_node("take_order", take_order)
graph_builder.add_node("confirm_order", confirm_order)

# Define conversation flow
graph_builder.add_edge(START, "take_order")
graph_builder.add_edge("take_order", "confirm_order")
graph_builder.add_edge("confirm_order", END)

# Compile chatbot
chatbot = graph_builder.compile()

# Simulate a conversation
test_input = "I want a burger."

messages = chatbot.invoke({"messages": [HumanMessage(content=test_input)]})

for message in messages["messages"]:
    print(f"Message: {message.content}")

print("Total Orders: ", messages["order_id"])
