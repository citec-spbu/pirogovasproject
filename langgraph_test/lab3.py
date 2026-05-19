import operator 
from langgraph.graph import START, END, StateGraph 
from typing import Annotated, List, Literal, TypedDict
from langgraph.types import Command, interrupt

class State(TypedDict):
    nlist: Annotated[list[str], operator.add]

def node_a(state) -> Command[Literal["b", "c", END]]:
    select = state["nlist"][-1]

    if select == "b":
        next_node = "b"
    
    elif select == "c":
        next_node = "c"

    elif select == "q":
        next_node = END

    else:
        next_node = END

    return Command(
            update = State(nlist = [select]),
            goto = [next_node]
    )

def node_b(state):
    return State(nlist = ["B"])

def node_c(state):
    return State(nlist = ["C"])

#def conditional_edge(state):
#    select = state["nlist"][-1]
    
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
#builder.add_conditional_edges("a", conditional_edge)

graph = builder.compile()

user = input()
initial_state = State(
        nlist = [user]
)


res = graph.invoke(initial_state)

from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

graph = builder.compile(checkpointer=memory)

while True:
    user = input()
    input_state = State(nlist = [user])
    res = graph.invoke(input_state, config)
    print(res)
    if res["nlist"][-1] == "q":
        print("quit")
        break












