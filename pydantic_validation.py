from pydantic import BaseModel
from langgraph.graph import END, START, StateGraph


class State(BaseModel):
    name: str


def greet(state: State) -> dict:
    greeting = f"Hello, {state.name}!"
    return {"name": greeting}


graph = StateGraph(State)
graph.add_node("greet", greet)
graph.add_edge(START, "greet")
graph.add_edge("greet", END)
runnable = graph.compile()
print(runnable.invoke(State(name="Alice")))
print(runnable.invoke(State(name=123)))
