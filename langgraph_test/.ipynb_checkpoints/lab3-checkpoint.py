import operator 
from langgraph.graph import START, END, StateGraph 
from typing import Annotated, List, Literal, TypedDict
from langgraph.types import Command, interrupt

class State(TypedDict):
    nlist: Annotated[list[str], operator.add]

def node_a(state):
    return 

def node_b(state):
    return State(nlist = ["B"])

def node_c(state):
    return State(nlist = ["C"])

def conditional_edge(state):
    select = state["nlist"][-1]
    
    if select == "b":
        return "b"

    elif select == "c":
        return "c"
    elif select == "q":
        return END
    
    else: 
        return END

builder = StateGraph(State)

builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)

builder.add_edge(START, "a")
builder.add_edge("b", END)
builder.add_edge("c", END)
builder.add_conditional_edges("a", conditional_edge)

graph = builder.compile()

user = input()
initial_state = State(
        nlist = [user]
)


res = graph.invoke(initial_state)

print(res)
