from IPython.display import Image, display
import operator 
from typing import Annotated, List, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

class State(TypedDict):
    nlist: List[str]

def node_a(state: State):
    print(state['nlist'])
    note = "Hi, I'm a"
    return(State(nlist=[note]))

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_edge(START, "a")
builder.add_edge("a", END)
graph = builder.compile()

img = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(img)
#print(graph.get_graph().draw_mermaid())

initial_state = State(
        nlist = ["Всем привет, сейчас я изучаю langgraph"]
        )
res = graph.invoke(initial_state)
print(res)
