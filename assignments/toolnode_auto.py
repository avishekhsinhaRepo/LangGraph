from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


@tool
def check_symptoms(symptom: str):
    """Provides possible conditions based on the symptom described."""
    conditions = {
        "fever": ["Flu", "COVID-19", "Common Cold"],
        "cough": ["Bronchitis", "Pneumonia", "Common Cold"],
        "headache": ["Migraine", "Tension Headache", "Sinus Infection"],
    }
    return conditions.get(symptom.lower(), ["No specific conditions found. Please consult a doctor."])


@tool
def book_doctor_appointment(specialty: str, date: str, time: str):
    """Books an appointment with a doctor based on the required specialty."""
    available_specialties = ["General Physician", "Cardiologist", "Neurologist", "Pediatrician"]
    if specialty in available_specialties:
        return f"Appointment booked with {specialty} on {date} at {time}."
    else:
        return f"Sorry, no available {specialty} at this time."


# Define tools
tools = [check_symptoms, book_doctor_appointment]


# Initialize the LLM
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)

# TODO: Create the ToolNode
tool_node = None


# TODO: Implement the Node
def call_model(state: MessagesState):
    pass


# TODO: Define Conditional Routing
def should_continue(state: MessagesState):
    pass


# ✅ Define the Workflow
workflow = StateGraph(MessagesState)

# TODO: Add Nodes


# ✅ Compile Workflow
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# ✅ Test the Workflow

config = {"configurable": {"thread_id": "1"}}

# ✅ Step 1: Check Symptoms
response = graph.invoke(
    {"messages": [HumanMessage(content="I have a fever. Can you tell me what this condition might be?")]},
    config
)

print(response["messages"][-1])
# ✅ Extract the conditions
conditions = response["messages"][-1].content
print("\n🔍 **Possible Conditions Based on Symptoms:**")
print(conditions)


# ✅ Step 2: Book Doctor Appointment
response = graph.invoke(
    {"messages": [HumanMessage(content="Book an appointment for these conditions"
                                       " with a General Physician for tomorrow at 10 AM.")]},
    config
)

# ✅ Extract the final response
final_response = response["messages"][-1].content

# ✅ Print the final response
print("\n📅 **Doctor Appointment Confirmation:**")
print(final_response)
