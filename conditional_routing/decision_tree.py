from typing import TypedDict
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

# Set your Azure OpenAI credentials
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


class DecisionMessage(TypedDict):
    message: str


def conditional_action(state: DecisionMessage):
    if "evening" in state["message"]:
        return "play"
    else:
        return "study"


def study(state: DecisionMessage):
    print(f"Message Recieved {state['message']}")
    return {"message": "Time To Play!!"}


def play(state: DecisionMessage):
    print(f"Message Recieved {state['message']}")
    return {"message": "Time To Study!!"}


graph = StateGraph(DecisionMessage)
graph.add_node("study", study)
graph.add_node("play", play)

graph.add_conditional_edges(START, conditional_action)
graph.add_edge("play", END)
graph.add_edge("study", END)

runnable = graph.compile()
output = runnable.invoke({"message": "Its evening time"})
print(f"Final Output: {output['message']}")
