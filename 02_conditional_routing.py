# 02_conditional_routing.py
# This script demonstrates how to use conditional edges to route the flow 
# based on dynamic decision making.

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Define State
class CustomerServiceState(TypedDict):
    query: str
    department: str
    resolution: str

# 2. Define Nodes
def intake_node(state: CustomerServiceState):
    print(f"[Intake] Analyzing query: '{state['query']}'")
    query_lower = state["query"].lower()
    
    # Simple rule-based routing logic (in a real app, an LLM would do this)
    if "refund" in query_lower or "money" in query_lower or "charge" in query_lower:
        dept = "billing"
    elif "password" in query_lower or "login" in query_lower or "bug" in query_lower:
        dept = "technical"
    else:
        dept = "general"
        
    # Update the department in the state
    return {"department": dept}

def billing_department(state: CustomerServiceState):
    print(f"[Billing] Handling billing query...")
    return {"resolution": "Processed refund or billing request."}

def technical_department(state: CustomerServiceState):
    print(f"[Tech] Handling technical query...")
    return {"resolution": "Reset password or logged a bug ticket."}

def general_department(state: CustomerServiceState):
    print(f"[General] Handling general query...")
    return {"resolution": "Answered general FAQ."}

# 3. Define Routing Function
# This function determines the NEXT node to go to.
def route_query(state: CustomerServiceState):
    # It returns a string that must map to a node name
    return state["department"]

# 4. Build Graph
workflow = StateGraph(CustomerServiceState)

workflow.add_node("intake", intake_node)
workflow.add_node("billing", billing_department)
workflow.add_node("technical", technical_department)
workflow.add_node("general", general_department)

# 5. Edges
workflow.add_edge(START, "intake")

# --- CONDITIONAL EDGE ---
# Arguments:
# 1. The source node ("intake")
# 2. The routing function that returns a value (route_query)
# 3. A dictionary mapping the return values of the routing function to the next node names
workflow.add_conditional_edges(
    "intake",
    route_query,
    {
        "billing": "billing",
        "technical": "technical",
        "general": "general"
    }
)

# All department nodes just go to the end
workflow.add_edge("billing", END)
workflow.add_edge("technical", END)
workflow.add_edge("general", END)

# 6. Compile
app = workflow.compile()

if __name__ == "__main__":
    test_queries = [
        "I was double charged for my subscription!",
        "I can't remember my login password.",
        "What are your business hours?"
    ]
    
    for q in test_queries:
        print("\n" + "="*40)
        # We start with just the query, other fields can be empty/None initially
        initial_state = {"query": q, "department": "", "resolution": ""}
        final_state = app.invoke(initial_state)
        print(f"Resulting Department: {final_state['department']}")
        print(f"Resolution: {final_state['resolution']}")
