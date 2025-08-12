from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import interrupt
from langgraph.types import Command


class CodingAssistantState(TypedDict):
    task: str
    code: str
    tests: str


model = ChatOpenAI()

code_prompt = ChatPromptTemplate.from_template("Generate Python code for: {task}")
test_prompt = ChatPromptTemplate.from_template("Write unit tests for this code:\n{code}")

code_chain = code_prompt | model | StrOutputParser()
test_chain = test_prompt | model | StrOutputParser()


def generate_code(state):
    print("Generate Code")
    code = code_chain.invoke({"task": state["task"]})
    return Command(goto="human_review", update={"code": code})


# TODO
def human_review(state):
    pass


def create_tests(state):
    tests = test_chain.invoke({"code": state["code"]})
    return Command(goto=END, update={"tests": tests})


def create_coding_assistant_workflow():
    workflow = StateGraph(CodingAssistantState)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("human_review", human_review)
    workflow.add_node("create_tests", create_tests)
    workflow.set_entry_point("generate_code")
    return workflow.compile(checkpointer=MemorySaver())


# Run the Workflow
coding_assistant = create_coding_assistant_workflow()
inputs = {"task": "Create a function to reverse a string in Python."}
thread = {"configurable": {"thread_id": 1}}
result = coding_assistant.invoke(inputs, config=thread)
# TODO: Handle Interrupt

print("\n--- Generated Tests ---")
print(result.get("tests", "No code or tests generated"))
