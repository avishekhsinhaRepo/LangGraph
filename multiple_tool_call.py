from langchain_core.messages import HumanMessage, AnyMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


load_dotenv()

# Set your Azure OpenAI credentials
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
tavily_api_key = os.environ["TAVILY_API_KEY"]


# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


######################### Tool Defination  Start ##################################
@tool
def add(op1: int, op2: int) -> int:
    """
    Adds two integers and returns their sum.
    Args:
        op1 (int): The first operand.
        op2 (int): The second operand.
    Returns:
        int: The sum of op1 and op2.
    """

    return op1 + op2


@tool
def multiply(op1: int, op2: int) -> int:
    """
    Multiplies two integers and returns the result.
    Args:
        op1 (int): The first operand.
        op2 (int): The second operand.
    Returns:
        int: The product of op1 and op2.
    """

    return op1 * op2


@tool
def wikipedia_search(query: str) -> str:
    """
    Searches Wikipedia for the given query.
    """
    wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper, verbose=True)
    return wiki_tool.invoke(query)  # Return the result directly


######################### Tool Defination  End ##################################

tools = [add, multiply, wikipedia_search]
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def llm_tool(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph = StateGraph(State)
graph.add_node("llm_tool", llm_tool)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "llm_tool")
graph.add_conditional_edges("llm_tool", tools_condition)
graph.add_edge("tools", END)

runnable = graph.compile()


messages = runnable.invoke({"messages": [HumanMessage("What is 5 + 3?")]})


messages = runnable.invoke({"messages": [HumanMessage("What is 5 * 3?")]})


messages = runnable.invoke({"messages": [HumanMessage("Who is Sachin Tendulkar?")]})

for message in messages["messages"]:
    message.pretty_print()
