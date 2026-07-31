import operator
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    next: str

def researcher(state: AgentState):
    print("Researcher is working...")
    return {"messages": ["Researcher: Found some data about AI."], "next": "writer"}

def writer(state: AgentState):
    print("Writer is working...")
    return {"messages": ["Writer: Drafted an article based on the data."], "next": "supervisor"}

def supervisor(state: AgentState):
    print("Supervisor is checking...")
    # Simple logic: If writer has written, we are done. Otherwise researcher.
    messages = state.get("messages", [])
    if any("Writer:" in m for m in messages):
        return {"next": "FINISH"}
    return {"next": "researcher"}

def router(state: AgentState):
    if state["next"] == "FINISH":
        return END
    return state["next"]

def main():
    builder = StateGraph(AgentState)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)
    builder.add_node("supervisor", supervisor)
    
    builder.add_edge(START, "supervisor")
    
    builder.add_conditional_edges(
        "supervisor",
        router,
        {"researcher": "researcher", "writer": "writer", END: END}
    )
    
    # Workers always go back to supervisor to decide what's next
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    
    graph = builder.compile()
    
    print("--- Multi-Agent Workflow Starting ---")
    final_state = graph.invoke({"messages": ["User: Write an article about AI."]})
    print("\n--- Final Messages ---")
    for msg in final_state["messages"]:
        print(msg)

if __name__ == "__main__":
    main()
