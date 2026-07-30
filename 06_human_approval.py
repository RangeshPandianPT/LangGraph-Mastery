import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    task: str
    is_approved: bool

def generate_task(state: State):
    print("--- Node: Generating Task ---")
    # In reality, an LLM might generate an action plan here
    return {"task": "Delete database tables", "is_approved": False}

def execute_task(state: State):
    print("--- Node: Executing Task ---")
    if state.get("is_approved"):
        print(f"EXECUTING DANGEROUS TASK: {state['task']}")
    else:
        print(f"TASK DENIED: {state['task']}")
    return state

builder = StateGraph(State)
builder.add_node("generate", generate_task)
builder.add_node("execute", execute_task)

builder.add_edge(START, "generate")
builder.add_edge("generate", "execute")
builder.add_edge("execute", END)

memory = MemorySaver()

# IMPORTANT: We add a breakpoint BEFORE the 'execute' node
graph = builder.compile(checkpointer=memory, interrupt_before=["execute"])

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "approval_thread_1"}}
    
    print("1. Starting graph...")
    # The graph will run 'generate' and then PAUSE before 'execute'
    for event in graph.stream({"task": "", "is_approved": False}, config=config):
        print(event)
        
    # Check the current state
    snapshot = graph.get_state(config)
    print("\n2. Graph paused! Current State:", snapshot.values)
    print("Next node to execute:", snapshot.next)
    
    # Simulate a human approving the action
    print("\n3. Human review... User approves the action.")
    
    # We can update the state manually as the human
    graph.update_state(config, {"is_approved": True})
    
    print("\n4. Resuming graph execution...")
    # Passing None resumes execution from the breakpoint
    for event in graph.stream(None, config=config):
        print(event)
