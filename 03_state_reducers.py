import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

# 1. Define the State with a Reducer
# Annotated[type, reducer_function] tells LangGraph how to update this key.
# operator.add means "append to the list" instead of "overwrite the list".
class State(TypedDict):
    messages: Annotated[list[str], operator.add]
    count: int # No reducer, so it will be overwritten each time

# 2. Define Nodes
def node_a(state: State):
    print("--- Node A ---")
    current_count = state.get("count", 0)
    # We return a list for messages. Because of operator.add, 
    # LangGraph will append this list to the existing messages list.
    return {"messages": ["Message from A"], "count": current_count + 1}

def node_b(state: State):
    print("--- Node B ---")
    current_count = state.get("count", 0)
    return {"messages": ["Message from B"], "count": current_count + 1}

# 3. Build Graph
builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)

builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

graph = builder.compile()

# 4. Run Graph
if __name__ == "__main__":
    print("Starting Graph with empty state...")
    # Initial state with a starting message and count 0
    initial_state = {"messages": ["Initial Message"], "count": 0}
    
    final_state = graph.invoke(initial_state)
    
    print("\n--- Final State ---")
    print(f"Messages (Appended): {final_state['messages']}")
    print(f"Count (Overwritten): {final_state['count']}")
