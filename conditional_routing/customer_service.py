from typing import TypedDict
from langgraph.graph import END, START, StateGraph


# Define the structure of the input state (customer support request)
class SupportRequest(TypedDict):
    message: str
    priority: int  # 1 (high), 2 (medium), 3 (low)


# Function to categorize the support request
def categorize_request(request: SupportRequest):
    # TODO: Implement Conditional Routing
    print(f"Received request: {request}")
    if request["priority"] == 1:
        return "handle_urgent"
    elif request["priority"] == 2:
        return "handle_standard"
    else:
        print("Low priority request, no action taken.")
        return END  # No further action for low priority requests
    
    

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
graph.add_node("handle_urgent", handle_urgent)
graph.add_node("handle_standard", handle_standard)


graph.add_conditional_edges(START, categorize_request)
graph.add_edge("handle_urgent", END)
graph.add_edge("handle_standard", END)

runnable = graph.compile()

# Simulate a customer support request
print(runnable.invoke({"message": "My account was hacked! Urgent help needed.", "priority": 1}))
print(runnable.invoke({"message": "I need help with password reset.", "priority": 3}))