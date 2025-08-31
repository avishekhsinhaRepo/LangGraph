from typing_extensions import TypedDict
from dataclasses import dataclass

# Import StateGraph from the appropriate module
from langgraph.graph import END, START, StateGraph


## State definition
@dataclass
class State:
    graph_info: str
    is_raining: bool


def start_play(state: State) -> dict:
    # Initialize the state
    return {"graph_info": state.graph_info + " I am planning to play."}


def cricket(state: State) -> dict:
    # Update the state for cricket
    return {"graph_info": state.graph_info + " cricket."}


def badminton(state: State) -> dict:
    # Update the state for badminton
    return {"graph_info": state.graph_info + " badminton."}


def play_condition(state: State) -> str:
    # Condition to decide which sport to play
    if state.is_raining:
        return "badminton"
    return "cricket"


graph = StateGraph(State)

graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)

graph.add_edge(START, "start_play")
graph.add_conditional_edges("start_play", play_condition)

graph.add_edge("cricket", END)
graph.add_edge("badminton", END)

runnable = graph.compile()
print(runnable.invoke(State(graph_info="", is_raining=False)))
