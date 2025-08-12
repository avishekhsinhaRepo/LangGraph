from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph


class HelloWorldState(BaseModel):
    message: str = Field(min_length=3, max_length=50)


def hello(state: HelloWorldState):
    print(f"Hello Node: {state.message}")
    return {"message": "Hello " + state.message}


def bye(state: HelloWorldState):
    print(f"Bye Node: {state.message}")
    return {"message": "Bye " + state.message}


graph = StateGraph(HelloWorldState)
graph.add_node("hello", hello)
graph.add_node("bye", bye)

graph.add_edge(START, "hello")
graph.add_edge("hello", "bye")
graph.add_edge("bye", END)

runnable = graph.compile()
output = runnable.invoke({"message": "World"})
print(f"Final Output: {output['message']}")
