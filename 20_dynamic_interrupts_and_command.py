"""
20_dynamic_interrupts_and_command.py: Modern Human-in-the-Loop with interrupt() and Command API

What you will learn:
1. Why LangGraph's dynamic `interrupt()` supersedes legacy static `interrupt_before`.
2. In-node pause: How to halt execution at an exact line of Python code based on dynamic runtime criteria.
3. Resuming execution: How to unblock the agent by invoking `Command(resume=...)`.
4. The `Command(update=..., goto=...)` primitive: Combining atomic state updates
   and dynamic node routing in a single return object without boilerplate edge functions.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# --- 1. Graph State ---
class CloudOpsState(TypedDict):
    operation: str
    resource: str
    estimated_cost: float
    human_approval: Optional[str]
    deployment_status: str

# --- 2. Node Functions ---

def analyze_request(state: CloudOpsState) -> Command:
    """
    Evaluates cloud operation request.
    Demonstrates Command(update=..., goto=...): Routes directly to appropriate node
    while atomically updating state, without requiring separate conditional edge routers!
    """
    op = state["operation"]
    cost = state["estimated_cost"]
    print(f"\n[OPS ANALYZER] Evaluating request: '{op}' on '{state['resource']}' (Estimated Cost: ${cost})")
    
    if cost > 500.0 or "delete" in op.lower() or "cluster" in op.lower():
        print("[OPS ANALYZER] Policy Alert: High-risk action detected. Routing to Human Gatekeeper...")
        return Command(
            update={"deployment_status": "FLAGGED_FOR_HUMAN_REVIEW"},
            goto="human_gatekeeper"
        )
    else:
        print("[OPS ANALYZER] Policy Check: Low risk. Fast-tracking to automated deployment...")
        return Command(
            update={"deployment_status": "AUTO_APPROVED"},
            goto="execute_deployment"
        )

def human_gatekeeper(state: CloudOpsState) -> dict:
    """
    Demonstrates dynamic in-node interrupt().
    Halts execution right here, yielding an interrupt payload to the client.
    When the user responds, execution resumes on the very next line with the return value!
    """
    op = state["operation"]
    cost = state["estimated_cost"]
    
    print(f"\n[HUMAN GATEKEEPER] Triggering dynamic interrupt() for high-risk action '${cost}'...")
    
    # Execution PAUSES here and returns payload to caller
    approval_decision = interrupt({
        "alert": "CRITICAL_ACTION_REQUIRING_SIGNOFF",
        "action": op,
        "resource": state["resource"],
        "cost": cost,
        "message": f"Do you authorize {op} on resource '{state['resource']}' costing ${cost}?"
    })
    
    # When resumed via Command(resume=...), execution continues right here!
    print(f"[HUMAN GATEKEEPER] Resumed! Human decision received: '{approval_decision}'")
    
    if approval_decision and approval_decision.upper() == "APPROVED":
        return {
            "human_approval": "APPROVED",
            "deployment_status": "MANUALLY_APPROVED"
        }
    else:
        return {
            "human_approval": "REJECTED",
            "deployment_status": "BLOCKED_BY_HUMAN"
        }

def execute_deployment(state: CloudOpsState) -> dict:
    """Executes the cloud operation if authorized."""
    status = state.get("deployment_status", "")
    op = state["operation"]
    resource = state["resource"]
    
    print(f"\n[DEPLOYMENT ENGINE] Processing action for '{resource}' with status: [{status}]")
    
    if "APPROVED" in status:
        final_msg = f"SUCCESS: '{op}' completed on '{resource}'. System is healthy."
    else:
        final_msg = f"ABORTED: '{op}' was rejected by policy or human reviewer. No changes applied."
        
    print(f"[DEPLOYMENT ENGINE] {final_msg}")
    return {"deployment_status": final_msg}

# --- 3. Build the Graph ---
def build_cloudops_graph():
    builder = StateGraph(CloudOpsState)
    
    builder.add_node("analyze_request", analyze_request)
    builder.add_node("human_gatekeeper", human_gatekeeper)
    builder.add_node("execute_deployment", execute_deployment)
    
    builder.add_edge(START, "analyze_request")
    # Notice: analyze_request uses Command(goto=...), so no explicit edges needed from it!
    
    builder.add_edge("human_gatekeeper", "execute_deployment")
    builder.add_edge("execute_deployment", END)
    
    # A checkpointer is MANDATORY for interrupt() so state can be saved and resumed
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

# --- 4. Main Demonstration ---
def main():
    print("=" * 70)
    print("LangGraph: Modern Dynamic Interrupts & Command API")
    print("=" * 70)
    
    app = build_cloudops_graph()
    
    # -----------------------------------------------------------------
    # SCENARIO 1: Low-Cost Request (Fast-paths without interruption)
    # -----------------------------------------------------------------
    print("\n" + "=" * 55)
    print(">>> SCENARIO 1: Automated Low-Risk Action (No Interruption) <<<")
    print("=" * 55)
    thread_1 = {"configurable": {"thread_id": "thread_low_risk"}}
    
    result_1 = app.invoke({
        "operation": "Scale ReplicaSet to 3 pods",
        "resource": "frontend-service",
        "estimated_cost": 25.0,
        "human_approval": None,
        "deployment_status": ""
    }, config=thread_1)
    
    print(f"Final Status: {result_1['deployment_status']}")
    
    # -----------------------------------------------------------------
    # SCENARIO 2: High-Cost Request (Dynamic Interrupt + Resumption)
    # -----------------------------------------------------------------
    print("\n" + "=" * 55)
    print(">>> SCENARIO 2: High-Risk Action (Dynamic Interrupt + Resume) <<<")
    print("=" * 55)
    thread_2 = {"configurable": {"thread_id": "thread_high_risk"}}
    
    print("\n[Step 1] Invoking graph with $4,500 GPU Cluster provisioning request...")
    app.invoke({
        "operation": "Provision 8x H100 GPU Cluster for Distributed Training",
        "resource": "cluster-deeplearning-prod",
        "estimated_cost": 4500.0,
        "human_approval": None,
        "deployment_status": ""
    }, config=thread_2)
    
    # Inspect interrupted state
    snapshot = app.get_state(thread_2)
    print(f"\n[Step 2] Execution paused! Is graph interrupted? -> {bool(snapshot.tasks)}")
    
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        interrupt_payload = snapshot.tasks[0].interrupts[0].value
        print("\n--- Human-in-the-Loop Interrupt Received ---")
        print(f"Alert:    {interrupt_payload.get('alert')}")
        print(f"Action:   {interrupt_payload.get('action')}")
        print(f"Cost:     ${interrupt_payload.get('cost')}")
        print(f"Prompt:   {interrupt_payload.get('message')}")
        
        # Simulate human operator signing off
        print("\n[Step 3] Human Operator reviews details and approves action...")
        print(">>> Submitting Command(resume='APPROVED')...")
        
        resumed_result = app.invoke(Command(resume="APPROVED"), config=thread_2)
        
        print("\n[Step 4] Workflow completed post-interruption:")
        print(f"Final Status: {resumed_result['deployment_status']}")

if __name__ == "__main__":
    main()
