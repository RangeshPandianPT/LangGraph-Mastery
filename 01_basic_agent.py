# 01_basic_agent.py
# This script demonstrates the absolute basics of LangGraph:
# State, Nodes, and Edges (Flow).
# We simulate a simple workflow without using real LLMs to focus purely on the graph mechanics.

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# 1. Define the State
# The state is a dictionary that gets passed between nodes. 
# Nodes will return a dictionary of updates that get applied to this state.
class AgentState(TypedDict):
    messages: List[str]
    status: str
    counter: int

# 2. Define Nodes
# Nodes are just Python functions that take the state as input and return state updates.
def node_alpha(state: AgentState):
    print("-> Executing Node Alpha")
    # We return a dictionary containing ONLY the keys we want to update.
    # For lists, by default LangGraph overrides them unless you configure them with `add_messages` or `operator.add`.
    # Here we just override it for simplicity.
    new_messages = state["messages"] + ["Alpha processed this."]
    return {
        "messages": new_messages, 
        "status": "in_progress",
        "counter": state["counter"] + 1
    }

def node_beta(state: AgentState):
    print("-> Executing Node Beta")
    new_messages = state["messages"] + ["Beta finished the job."]
    return {
        "messages": new_messages, 
        "status": "completed",
        "counter": state["counter"] + 1
    }

# 3. Create the Graph Builder
workflow = StateGraph(AgentState)

# 4. Add Nodes to the Graph
# The first argument is the string name of the node, the second is the function.
workflow.add_node("node_a", node_alpha)
workflow.add_node("node_b", node_beta)

# 5. Define Edges (The Flow)
# The graph always starts at the special `START` node.
workflow.add_edge(START, "node_a")
# After node_a, go to node_b
workflow.add_edge("node_a", "node_b")
# After node_b, end the graph execution.
workflow.add_edge("node_b", END)

# 6. Compile the Graph
# Compiling turns the abstract builder into an executable app.
app = workflow.compile()

if __name__ == "__main__":
    print("Starting the simple LangGraph workflow...\n")
    
    # Define our starting state
    initial_state = {
        "messages": ["User: Hello!"],
        "status": "pending",
        "counter": 0
    }
    
    # Run the graph using `invoke`
    final_state = app.invoke(initial_state)
    
    print("\n--- Final State ---")
    import json
    print(json.dumps(final_state, indent=2))
    print("Notice how the 'messages', 'status', and 'counter' were updated by the nodes!")
