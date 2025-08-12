from langgraph.graph import StateGraph, END, START
from typing import TypedDict


# Define Shared State
class InsuranceState(TypedDict):
    patient_id: str
    insurance_verified: bool


def verify_insurance_check(state: InsuranceState):
    print("verify_insurance_check")
    if state["patient_id"] is not None:
        return {"insurance_verified": True, "appointment_status": "Insurance verification in progress"}
    else:
        return {"insurance_verified": False, "appointment_status": "Insurance verification pending"}


def verify_insurance_confirm(state: InsuranceState):
    print("verify_insurance_confirm")
    if state["insurance_verified"]:
        return {"appointment_status": "Insurance verified"}
    else:
        return {"appointment_status": "Insurance verification failed"}


# Insurance Verification Subgraph
insurance_graph = StateGraph(InsuranceState)
insurance_graph.add_node("verify_insurance_check", verify_insurance_check)
insurance_graph.add_node("verify_insurance_confirm", verify_insurance_confirm)
insurance_graph.add_edge(START, "verify_insurance_check")
insurance_graph.add_edge("verify_insurance_check", "verify_insurance_confirm")
insurance_graph.add_edge("verify_insurance_confirm", END)
insurance_graph = insurance_graph.compile()


# Define Shared State
class AppointmentState(TypedDict):
    patient_id: str
    appointment_status: str
    insurance_verified: bool
    appointment_scheduled: bool


def schedule_appointment(state: AppointmentState):
    print("schedule_appointment")
    if state["insurance_verified"]:
        return {"appointment_scheduled": True, "appointment_status": "Appointment scheduled"}
    else:
        return {"appointment_scheduled": False, "appointment_status": "Appointment scheduling failed: Insurance issue"}


# Main Appointment Management Graph
appointment_graph = StateGraph(AppointmentState)
# TODO: Add Sub Graph as a node
appointment_graph.add_node("schedule_appointment", schedule_appointment)

# Define edges
# TODO: Add the subgraph edge
appointment_graph.add_edge("insurance_verification", "schedule_appointment")
appointment_graph.add_edge("schedule_appointment", END)

appointment_graph = appointment_graph.compile()

# Invoke main workflow
inputs = {
    "patient_id": "PT-2025",
}
output = appointment_graph.invoke(inputs)

# TODO: Print the final output
print(output)