import chainlit as cl
from claim_processing_agent import create_workflow  # Replace with your actual graph module
from langgraph.types import Command

graph = create_workflow()

conversation_stage = "get_patient_id"
claim_info = {}
thread = {"configurable": {"thread_id": "101"}}


@cl.on_chat_start
async def on_start():
    global conversation_stage
    conversation_stage = "get_patient_id"
    await cl.Message("Welcome to the 💼 Claim Validation Assistant!").send()
    await cl.Message("🔹 Please enter the **Patient ID**:").send()


@cl.on_message
async def handle_message(message):
    global conversation_stage, claim_info

    if conversation_stage == "get_patient_id":
        claim_info["patient_id"] = message.content.strip()
        await cl.Message("🔹 Enter the **Treatment Code** (e.g., Z12.31):").send()
        conversation_stage = "get_treatment_code"
        return

    if conversation_stage == "get_treatment_code":
        claim_info["treatment_code"] = message.content.strip()
        await cl.Message("🔹 Enter the **Claim Details**:").send()
        conversation_stage = "get_claim_details"
        return

    if conversation_stage == "get_claim_details":
        claim_info["claim_details"] = message.content.strip()
        await cl.Message("🧰 Validating claim... Please wait.").send()

        state = graph.invoke(claim_info, config=thread)
        # TODO: Human In The Loop

        # No interrupt, display results directly
        await show_results(state)
        conversation_stage = "restart"
        return

    if conversation_stage == "await_approval":
        # TODO: Human In The Loop
        pass
    
    if conversation_stage == "restart" and message.content.strip().lower() == "restart":
        claim_info = {}
        conversation_stage = "get_patient_id"
        await cl.Message("🔄 Restarting process... Please enter the **Patient ID**:").send()
        return

    await cl.Message("⚠️ I did not understand that. Please follow the instructions.").send()


async def show_results(state):
    await cl.Message("\ud83d\udcc4 **Claim Summary:**").send()
    await cl.Message(f"\U0001f464 Patient Data:\n{state['patient_data']}").send()
    await cl.Message(f"\U0001f4b3 Insurance Data:\n{state['insurance_data']}").send()
    await cl.Message(f"\U0001f4dc Policy Docs:\n{state['policy_docs']}").send()
    await cl.Message(f"\U0001f916 AI Feedback:\n{state['ai_validation_feedback']}").send()
    await cl.Message(f"\ud83d\udccc Final Decision: **{state['final_decision']}**").send()
    await cl.Message("\ud83d\udd04 Type `restart` to process another claim.").send()