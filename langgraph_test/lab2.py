from IPython.display import Image, display
import operator 
from typing import Annotated, List, Literal, TypedDict 
from langgraph.graph import END, START, StateGraph 
from langgraph.types import Command, interrupt 

class State(TypedDict):
    nlist: Annotated[List[str], operator.add] # Here operator.add is a reducer func.

def node_a(state: State) -> State: 
    print(f"Adding 'A' to {state['nlist']}")
    return (State(nlist = ["A"]))

def node_b(state: State):
    print(f"Adding 'B' to {state['nlist']}")
    return (State(nlist = ["B"]))

def node_c(state: State):
    print(f"Adding 'C' to {state['nlist']}")
    return (State(nlist = ["C"]))

def node_bb(state: State):
    print(f"Adding 'BB' to {state['nlist']}")
    return State(nlist = ["BB"])

def node_cc(state: State):
    print(f"Adding 'CC' to {state['nlist']}")
    return State(nlist = ["CC"])

def node_d(state: State):
    print(f"Adding 'D' to {state['nlist']}")
    return State(nlist = ["D"])

builder = StateGraph(State)

builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("bb", node_bb)
builder.add_node("cc", node_cc)
builder.add_node("d", node_d)
builder.add_node("c", node_c)

builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "bb")
builder.add_edge("c", "cc")
builder.add_edge("cc", "d")
builder.add_edge("bb", "d")
builder.add_edge("d", END)

graph = builder.compile()

initial_state = State(
        nlist = ['Initial sting:']
)
res = graph.invoke(initial_state)

print(res)

