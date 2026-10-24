"""
16_plan_and_execute.py: Plan-and-Execute (Solve) Architecture in LangGraph

What you will learn:
1. Why traditional ReAct agents can wander or fail on multi-step reasoning.
2. How the Plan-and-Execute pattern separates planning from execution.
3. Building a 3-node architecture:
   - Planner: Breaks a complex goal into an ordered list of tasks.
   - Executor: Executes the active task using tools or LLM reasoning.
   - Replanner: Evaluates execution output, determines if the goal is met,
     and dynamically adjusts/removes remaining tasks or emits the final response.
"""

import os
import operator
from typing import TypedDict, Annotated, List, Tuple, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# --- 1. Structured Schemas for Planning ---
class Plan(BaseModel):
    """List of steps to execute in sequence."""
    steps: List[str] = Field(description="Step-by-step tasks to achieve the user's objective")

class ReplannerOutput(BaseModel):
    """Output from the replanner."""
    is_complete: bool = Field(description="True if all steps are completed and final answer is ready")
    remaining_steps: List[str] = Field(default_factory=list, description="Updated list of remaining steps if not complete")
    final_response: Optional[str] = Field(default=None, description="Final response for the user if complete")

# --- 2. Graph State ---
class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: Optional[str]

# --- 3. Mock Execution Tools (Simulates real execution) ---
def execute_step_logic(task: str) -> str:
    """Mock execution engine for demonstration."""
    task_lower = task.lower()
    if "market size" in task_lower or "research" in task_lower:
        return "Market research data: Global enterprise AI market is estimated at $45B in 2024, growing at 32% CAGR."
    elif "competitors" in task_lower or "players" in task_lower:
        return "Competitor data: Key players include OpenAI, Anthropic, Google Cloud, and AWS Bedrock."
    elif "swot" in task_lower or "analysis" in task_lower:
        return "SWOT analysis: Strengths: High demand. Weaknesses: High compute costs. Opportunities: Multi-agent automation. Threats: Model commoditization."
    elif "draft" in task_lower or "executive summary" in task_lower:
        return "Executive summary drafted based on market size ($45B), key players, and compute cost challenges."
    return f"Completed step '{task}' successfully with verified analytical findings."

# --- 4. Node Definitions ---
def planner_node(state: PlanExecuteState) -> dict:
    """Creates the initial step-by-step plan."""
    print("\n[PLANNER] Analyzing objective and generating execution plan...")
    user_input = state["input"]
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            structured_llm = llm.with_structured_output(Plan)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert strategic planner. Create a concise, 3-to-4 step plan to solve the user's request."),
                ("human", "Objective: {input}")
            ])
            plan = (prompt | structured_llm).invoke({"input": user_input})
            steps = plan.steps
        except Exception as e:
            print(f"[NOTE] LLM Planner fallback triggered: {e}")
            steps = [
                f"Research market size and growth trends for {user_input}",
                f"Identify top competitive players for {user_input}",
                f"Synthesize findings into an executive report for {user_input}"
            ]
    else:
        steps = [
            f"Research market size and growth trends for {user_input}",
            f"Identify top competitive players for {user_input}",
            f"Synthesize findings into an executive report for {user_input}"
        ]
        
    print(f"Generated {len(steps)} steps:")
    for idx, s in enumerate(steps, 1):
        print(f"   {idx}. {s}")
        
    return {"plan": steps}

def executor_node(state: PlanExecuteState) -> dict:
    """Executes the next pending step from the plan."""
    current_step = state["plan"][0]
    print(f"\n[EXECUTOR] Executing step: '{current_step}'")
    
    # Run tool / task execution
    result = execute_step_logic(current_step)
    print(f"Result: {result[:80]}...")
    
    return {"past_steps": [(current_step, result)]}

def replanner_node(state: PlanExecuteState) -> dict:
    """Reviews progress and decides whether to update steps or finish."""
    print("\n[REPLANNER] Reviewing past steps and remaining plan...")
    current_plan = state["plan"][1:]  # remove the step just executed
    past_steps = state["past_steps"]
    
    # If no more steps remain, we finish
    if not current_plan:
        print("[REPLANNER] All steps completed! Assembling final synthesis...")
        synthesis = "\n".join([f"Step: {step}\nResult: {out}\n" for step, out in past_steps])
        final_answer = (
            f"--- Final Executive Report on '{state['input']}' ---\n"
            f"Execution Summary:\n{synthesis}\n"
            f"Recommendation: Proceed with enterprise multi-agent deployment focusing on high ROI workflows."
        )
        return {"plan": [], "response": final_answer}
    
    # Otherwise, update plan with remaining steps
    print(f"[REPLANNER] {len(current_plan)} steps remaining in the active plan.")
    return {"plan": current_plan}

# --- 5. Conditional Edge Router ---
def should_continue(state: PlanExecuteState):
    """Routes to END if final response is ready, otherwise repeats executor."""
    if state.get("response"):
        return END
    return "executor"

# --- 6. Build the Graph ---
def create_plan_execute_graph():
    builder = StateGraph(PlanExecuteState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("replanner", replanner_node)
    
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "replanner")
    
    builder.add_conditional_edges(
        "replanner",
        should_continue,
        {
            "executor": "executor",
            END: END
        }
    )
    
    return builder.compile()

# --- 7. Main Execution ---
def main():
    print("=" * 60)
    print("LangGraph: Plan-and-Execute (Solve) Architecture")
    print("=" * 60)
    
    graph = create_plan_execute_graph()
    
    user_goal = "Enterprise Agentic AI Market Strategy"
    initial_state = {
        "input": user_goal,
        "plan": [],
        "past_steps": [],
        "response": None
    }
    
    result = graph.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("FINAL AGENT OUTPUT:")
    print("=" * 60)
    print(result["response"])

if __name__ == "__main__":
    main()
