from langchain_core.messages import HumanMessage, AnyMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict

load_dotenv()

# Set your Azure OpenAI credentials
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def add(num1: int, num2: int) -> int:
    """
    Adds two integers and returns their sum.
    Args:
        num1 (int): The first integer to add.
        num2 (int): The second integer to add.
    Returns:
        int: The sum of num1 and num2.
    """

    return num1 + num2


# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

tools = [add]

# Binding Tools with LLM
llm_with_tools = llm.bind_tools(tools)


def llm_tool(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph = StateGraph(State)

graph.add_node("llm_tool", llm_tool)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "llm_tool")
graph.add_conditional_edges(
    "llm_tool",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
graph.add_edge("tools", END)

runnable = graph.compile()

messages = runnable.invoke({"messages": [HumanMessage("What is 5 + 3?")]})


for message in messages["messages"]:
    message.pretty_print()
