import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    step: str
    count: int

def node_a(state: State):
    time.sleep(0.5) # Simulate work
    return {"step": "Node A done", "count": state.get("count", 0) + 1}

def node_b(state: State):
    time.sleep(0.5)
    return {"step": "Node B done", "count": state["count"] + 1}

def node_c(state: State):
    time.sleep(0.5)
    return {"step": "Node C done", "count": state["count"] + 1}

def main():
    builder = StateGraph(State)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_node("c", node_c)
    
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", END)
    
    graph = builder.compile()
    
    print("--- Streaming Updates ---")
    # Stream mode "updates" yields the state updates from each node as they finish
    for update in graph.stream({"count": 0}, stream_mode="updates"):
        for node_name, state_update in update.items():
            print(f"[{node_name}] produced update: {state_update}")
            
    print("\n--- Streaming Values ---")
    # Stream mode "values" yields the full state after each step
    for full_state in graph.stream({"count": 0}, stream_mode="values"):
        print(f"Current full state: {full_state}")

if __name__ == "__main__":
    main()
