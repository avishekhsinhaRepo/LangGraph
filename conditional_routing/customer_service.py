from typing import TypedDict
from langgraph.graph import END, START, StateGraph


# Define the structure of the input state (customer support request)
class SupportRequest(TypedDict):
    message: str
    priority: int  # 1 (high), 2 (medium), 3 (low)


# Function to categorize the support request
def categorize_request(request: SupportRequest):
    print(f"Received request: {request}")
    # TODO: Implement Conditional Routing
    

# Function to process high-priority requests
def handle_urgent(request: SupportRequest):
    print(f"Routing to Urgent Support Team: {request}")
    return request


# Function to process standard requests
def handle_standard(request: SupportRequest):
    print(f"Routing to Standard Support Queue: {request}")
    return request


# Create the state graph
graph = StateGraph(SupportRequest)
# TODO: Create the graph


runnable = graph.compile()

# Simulate a customer support request
print(runnable.invoke({"message": "My account was hacked! Urgent help needed.", "priority": 1}))
print(runnable.invoke({"message": "I need help with password reset.", "priority": 3}))