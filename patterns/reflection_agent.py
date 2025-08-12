from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from typing import TypedDict
from langgraph.graph import END
from langgraph.types import Command

# Initialize OpenAI model
llm = ChatOpenAI()


# Define Agent State
class CodeState(TypedDict):
    problem_statement: str
    generated_code: str
    review_feedback: str
    refined_code: str
    iteration: int
    review_score: float


# 🟢 Step 1: Generate Initial Code
def generate_code(state: CodeState):
    print("Generating Code")
    prompt = f"""
    Write a clean, efficient, and well-commented Python solution for the following problem:

    {state['problem_statement']}
    """
    response = llm.invoke([SystemMessage(content="You are an expert Python developer."), HumanMessage(content=prompt)])

    state["generated_code"] = response.content
    state["iteration"] = 1
    state["review_score"] = 0
    return Command(goto="review_code", update=state)


# 🟢 Step 2: Review the Generated Code with a Score
def review_code(state: CodeState):
    print("Reviewing Code")
    prompt = f"""
    Review the following Python code for correctness, readability, efficiency, and best practices:

    {state['generated_code']}

    Provide a list of improvements and necessary changes.
    Also, give a **review score** (1-10) for:
    - Correctness
    - Readability
    - Efficiency
    - Maintainability

    Provide the average score out of 10 at the end.
    The last line should contain just the final score as a final_score:score
    """
    response = llm.invoke(
        [SystemMessage(content="You are a senior software engineer reviewing code."), HumanMessage(content=prompt)]
    )

    state["review_feedback"] = response.content

    # Extract the review score from AI feedback (assuming last line is "final_score: 8")
    try:
        lines = response.content.split("\n")
        last_line = lines[-1]
        state["review_score"] = float(last_line.split(":")[-1].strip())
    except:
        print("EXCEPT")
        state["review_score"] = 5  # Default if parsing fails

    return Command(goto="improve_code", update=state)


# 🟢 Step 3: Improve Code Based on Feedback
def improve_code(state: CodeState):
    print("Improving Code")
    print("Review Score ", state["review_score"], "Iteration ", state["iteration"])
    # TODO: Stop Iteration

    prompt = f"""
    Here is the initial Python code:

    {state['generated_code']}

    And here is the review feedback:

    {state['review_feedback']}

    Apply the suggested improvements and rewrite the code with better efficiency, readability, and correctness.
    """
    response = llm.invoke([SystemMessage(content="You are an AI code refiner."), HumanMessage(content=prompt)])

    state["generated_code"] = response.content
    state["iteration"] += 1
    return Command(goto="review_code", update=state)


# 🔵 Build the LangGraph Workflow
workflow = StateGraph(CodeState)

# Adding Nodes
workflow.add_node("generate_code", generate_code)
workflow.add_node("review_code", review_code)
workflow.add_node("improve_code", improve_code)

# Define Execution Flow
workflow.set_entry_point("generate_code")

# Compile Graph
graph = workflow.compile()

# 🟢 Run Example
input_data = {
    "problem_statement": "Write a function to find the factorial of a number in Python."
}

result = graph.invoke(input_data)

print("🚀 Final Code After Reflection:\n", result["generated_code"])
print("\n🔍 Final Review Feedback:\n", result["review_feedback"])