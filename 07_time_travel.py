import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list, operator.add]

def agent(state: State):
    return {"messages": ["Agent processed: " + state["messages"][-1]]}

def build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "1"}}
    
    print("--- 1. Initial Run ---")
    graph.invoke({"messages": ["Hello!"]}, config)
    graph.invoke({"messages": ["How are you?"]}, config)
    
    print("\n--- 2. Viewing State History ---")
    history = list(graph.get_state_history(config))
    # History is returned newest first.
    for i, state_snapshot in enumerate(history):
        print(f"Step {len(history) - i - 1}: {state_snapshot.values['messages']}")

    print("\n--- 3. Time Travel (Reverting State) ---")
    # Let's say we want to go back to the first step
    old_state = history[-2] 
    
    # We can invoke from this past configuration to fork history
    # The config of the past state has a unique checkpoint_id
    past_config = old_state.config
    print(f"Re-running from past config: {past_config}")
    new_result = graph.invoke({"messages": ["Wait, I meant: Hi!"]}, past_config)
    print("New result after time travel fork:", new_result)

if __name__ == "__main__":
    main()
