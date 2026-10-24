"""
21_reflection_and_self_critique.py: Reflexion & Self-Correction Architecture

What you will learn:
1. The Reflexion Pattern (Actor - Tester - Reflector loop).
2. Why one-shot LLM generations fail on edge cases and how self-critique repairs them.
3. Building an autonomous self-correcting agent:
   - Generator (Actor): Produces candidate code/logic conditioned on accumulated reflections.
   - Test Runner (Environment): Executes automated test suites and captures failures/traces.
   - Reflector (Critic): Diagnoses root cause of failures, writes verbal lessons into state.
   - Conditional Loop: Repeats generation until tests pass or max iterations are reached.
"""

import operator
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, START, END

# --- 1. Graph State ---
class ReflexionState(TypedDict):
    task_spec: str
    code_draft: str
    test_results: dict
    reflections: Annotated[List[str], operator.add]
    iteration: int
    max_iterations: int
    is_success: bool

# --- 2. Candidate Code Implementations (Simulating Iterative Self-Correction) ---
# Iteration 1 draft: Naive implementation that fails edge cases (ZeroDivisionError & missing keys)
DRAFT_V1 = """
def aggregate_service_metrics(logs: list) -> dict:
    total_latency = 0
    error_count = 0
    for entry in logs:
        total_latency += entry['latency_ms']
        if entry['status'] >= 400:
            error_count += 1
    return {
        'avg_latency': total_latency / len(logs),
        'total_requests': len(logs),
        'error_rate': error_count / len(logs)
    }
"""

# Iteration 2 draft: Improved implementation guided by the reflector's critique
DRAFT_V2 = """
def aggregate_service_metrics(logs: list) -> dict:
    if not logs:
        return {'avg_latency': 0.0, 'total_requests': 0, 'error_rate': 0.0}
    valid_entries = [e for e in logs if isinstance(e, dict) and 'latency_ms' in e and 'status' in e]
    if not valid_entries:
        return {'avg_latency': 0.0, 'total_requests': 0, 'error_rate': 0.0}
    total_latency = sum(e['latency_ms'] for e in valid_entries)
    error_count = sum(1 for e in valid_entries if e['status'] >= 400)
    count = len(valid_entries)
    return {
        'avg_latency': round(total_latency / count, 2),
        'total_requests': count,
        'error_rate': round(error_count / count, 4)
    }
"""

# --- 3. Node Functions ---

def generator_node(state: ReflexionState) -> dict:
    """Actor Node: Generates candidate code informed by past reflections."""
    current_iter = state["iteration"] + 1
    past_reflections = state.get("reflections", [])
    
    print(f"\n[GENERATOR (ACTOR)] Iteration {current_iter}/{state['max_iterations']}")
    if past_reflections:
        print(f"Incorporating {len(past_reflections)} past reflection(s):")
        for r in past_reflections:
            print(f"   * {r}")
        code = DRAFT_V2
    else:
        print("Generating initial candidate solution...")
        code = DRAFT_V1
        
    return {
        "code_draft": code.strip(),
        "iteration": current_iter
    }

def test_runner_node(state: ReflexionState) -> dict:
    """Environment Node: Executes automated unit tests against the code draft."""
    code = state["code_draft"]
    print("\n[TEST RUNNER (ENVIRONMENT)] Executing automated test suite in sandbox...")
    
    # Dynamic execution namespace
    exec_scope = {}
    try:
        exec(code, exec_scope)
        fn = exec_scope.get("aggregate_service_metrics")
        if not fn:
            raise ValueError("Function 'aggregate_service_metrics' not found in namespace")
    except Exception as e:
        return {
            "test_results": {"passed": False, "error": f"Compilation Error: {str(e)}"},
            "is_success": False
        }
        
    # Test suite cases
    test_cases = [
        # Test 1: Standard happy path
        ([{"latency_ms": 100, "status": 200}, {"latency_ms": 200, "status": 500}], 
         {"expected_count": 2, "expected_avg": 150.0}),
        # Test 2: Edge case: Empty input list
        ([], {"expected_count": 0, "expected_avg": 0.0}),
        # Test 3: Edge case: Corrupted entries with missing keys
        ([{"corrupted": True}, {"latency_ms": 50, "status": 200}], 
         {"expected_count": 1, "expected_avg": 50.0})
    ]
    
    for idx, (input_data, expectations) in enumerate(test_cases, 1):
        try:
            res = fn(input_data)
            if res["total_requests"] != expectations["expected_count"]:
                return {
                    "test_results": {
                        "passed": False,
                        "failed_test": f"Test {idx} (Expected count {expectations['expected_count']}, got {res['total_requests']})",
                        "error": "Count mismatch on filtered data"
                    },
                    "is_success": False
                }
            if res["avg_latency"] != expectations["expected_avg"]:
                return {
                    "test_results": {
                        "passed": False,
                        "failed_test": f"Test {idx} (Expected avg {expectations['expected_avg']}, got {res['avg_latency']})",
                        "error": "Latency calculation mismatch"
                    },
                    "is_success": False
                }
        except Exception as e:
            return {
                "test_results": {
                    "passed": False,
                    "failed_test": f"Test {idx} (Input: {input_data})",
                    "error": f"{type(e).__name__}: {str(e)}"
                },
                "is_success": False
            }
            
    print("All 3 test cases PASSED successfully!")
    return {
        "test_results": {"passed": True, "error": None},
        "is_success": True
    }

def reflector_node(state: ReflexionState) -> dict:
    """Critic/Reflector Node: Analyzes test failure and formulates verbal self-critique."""
    failure = state["test_results"]
    print("\n[REFLECTOR (CRITIC)] Diagnosing root cause of failure...")
    
    error_msg = failure.get("error", "Unknown error")
    failed_test = failure.get("failed_test", "N/A")
    print(f"Failed at: {failed_test}")
    print(f"Captured Error: {error_msg}")
    
    reflection = (
        f"Failure diagnosis at {failed_test}: "
        f"The code raised '{error_msg}'. Cause: unvalidated dictionary access and unguarded division by zero. "
        f"Actionable fix: Guard against empty log lists upfront and validate 'latency_ms' key existence before summing."
    )
    print(f"Formulated Reflection: {reflection}")
    return {"reflections": [reflection]}

# --- 4. Conditional Edge Router ---
def should_continue(state: ReflexionState) -> str:
    if state["is_success"]:
        print("\n>>> Self-Correction Goal Achieved: Solution Verified! <<<")
        return END
    if state["iteration"] >= state["max_iterations"]:
        print("\n>>> Max iterations reached without full pass. Terminating. <<<")
        return END
    return "reflector"

# --- 5. Build Graph ---
def build_reflexion_graph():
    builder = StateGraph(ReflexionState)
    
    builder.add_node("generator", generator_node)
    builder.add_node("test_runner", test_runner_node)
    builder.add_node("reflector", reflector_node)
    
    builder.add_edge(START, "generator")
    builder.add_edge("generator", "test_runner")
    
    builder.add_conditional_edges(
        "test_runner",
        should_continue,
        {
            "reflector": "reflector",
            END: END
        }
    )
    
    builder.add_edge("reflector", "generator")
    
    return builder.compile()

# --- 6. Main Execution ---
def main():
    print("=" * 70)
    print("LangGraph: Reflexion Architecture (Actor-Tester-Reflector Self-Correction)")
    print("=" * 70)
    
    app = build_reflexion_graph()
    
    initial_state = {
        "task_spec": "Write a robust aggregate_service_metrics(logs) function handling empty & malformed inputs",
        "code_draft": "",
        "test_results": {},
        "reflections": [],
        "iteration": 0,
        "max_iterations": 3,
        "is_success": False
    }
    
    final_state = app.invoke(initial_state)
    
    print("\n" + "=" * 55)
    print("FINAL VERIFIED CODE ARTIFACT:")
    print("=" * 55)
    print(final_state["code_draft"])
    print(f"\nFinal Success Status: {final_state['is_success']}")
    print(f"Total Iterations:     {final_state['iteration']}")
    print(f"Reflections Applied:  {len(final_state['reflections'])}")

if __name__ == "__main__":
    main()
