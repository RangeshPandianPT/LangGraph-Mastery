"""
17_hierarchical_subgraphs.py: Hierarchical Multi-Agent Systems with Nested Subgraphs

What you will learn:
1. The architectural need for Subgraphs: State isolation and modular agent domains.
2. How to define independent child graphs with their own private TypedDict schemas.
3. How to compile child graphs and insert them directly as nodes inside a parent orchestrator graph.
4. Clean state mapping: How data flows from parent -> child -> parent without state schema collisions.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. SUBGRAPH A: Research Agent (Specialized Domain & Private State)
# =====================================================================
class ResearchState(TypedDict):
    topic: str
    sources_found: List[str]
    research_summary: str

def fetch_sources_node(state: ResearchState) -> dict:
    topic = state["topic"]
    print(f"  [Subgraph: Research] Fetching academic and technical sources for: '{topic}'...")
    sources = [
        f"Whitepaper: High-performance state persistence in distributed systems ({topic})",
        f"Benchmark Study: Throughput comparison of async graph runtimes ({topic})"
    ]
    return {"sources_found": sources}

def synthesize_research_node(state: ResearchState) -> dict:
    print(f"  [Subgraph: Research] Synthesizing findings across {len(state['sources_found'])} sources...")
    summary = (
        f"Research Brief for '{state['topic']}':\n"
        + "\n".join([f"  * {s}" for s in state["sources_found"]])
        + "\nKey Insight: Subgraph modularity reduces memory footprint by 40%."
    )
    return {"research_summary": summary}

def build_research_subgraph():
    builder = StateGraph(ResearchState)
    builder.add_node("fetch_sources", fetch_sources_node)
    builder.add_node("synthesize", synthesize_research_node)
    
    builder.add_edge(START, "fetch_sources")
    builder.add_edge("fetch_sources", "synthesize")
    builder.add_edge("synthesize", END)
    
    return builder.compile()


# =====================================================================
# 2. SUBGRAPH B: Code Development Agent (Specialized Domain & Private State)
# =====================================================================
class CodeState(TypedDict):
    specification: str
    raw_code: str
    review_status: str

def generate_code_node(state: CodeState) -> dict:
    spec = state["specification"]
    print(f"  [Subgraph: Code] Generating implementation according to spec: '{spec}'...")
    code = (
        "def execute_pipeline(data: list) -> dict:\n"
        "    \"\"\"Production batch execution pipeline.\"\"\"\n"
        "    transformed = [x.strip().upper() for x in data if x]\n"
        "    return {\"status\": \"success\", \"count\": len(transformed), \"items\": transformed}"
    )
    return {"raw_code": code}

def review_code_node(state: CodeState) -> dict:
    print("  [Subgraph: Code] Reviewing code against security and style guidelines...")
    # Simulated lint and safety check
    status = "APPROVED: 0 lint errors, type checks passed (mypy compatible)."
    return {"review_status": status}

def build_code_subgraph():
    builder = StateGraph(CodeState)
    builder.add_node("generate_code", generate_code_node)
    builder.add_node("review_code", review_code_node)
    
    builder.add_edge(START, "generate_code")
    builder.add_edge("generate_code", "review_code")
    builder.add_edge("review_code", END)
    
    return builder.compile()


# =====================================================================
# 3. PARENT ORCHESTRATOR GRAPH
# =====================================================================
class ParentState(TypedDict):
    request: str
    target_domain: str
    research_report: str
    code_artifact: str
    final_delivery: str

# Compiled subgraphs to be invoked as nodes
research_subgraph = build_research_subgraph()
code_subgraph = build_code_subgraph()

def supervisor_router_node(state: ParentState) -> dict:
    print(f"\n[Supervisor] Analyzing request: '{state['request']}'")
    req = state["request"].lower()
    if "code" in req or "pipeline" in req or "function" in req:
        domain = "coding"
    else:
        domain = "research"
    print(f"[Supervisor] Routing workflow to specialized domain: '{domain.upper()}'")
    return {"target_domain": domain}

def call_research_subgraph_node(state: ParentState) -> dict:
    print("\n--- Invoking Research Subgraph ---")
    # State transformation: Map parent state to child subgraph state
    child_input: ResearchState = {
        "topic": state["request"],
        "sources_found": [],
        "research_summary": ""
    }
    child_output = research_subgraph.invoke(child_input)
    print("--- Research Subgraph Complete ---\n")
    # Return updates to parent state
    return {"research_report": child_output["research_summary"]}

def call_code_subgraph_node(state: ParentState) -> dict:
    print("\n--- Invoking Code Subgraph ---")
    # State transformation: Map parent state to child subgraph state
    child_input: CodeState = {
        "specification": state["request"],
        "raw_code": "",
        "review_status": ""
    }
    child_output = code_subgraph.invoke(child_input)
    print("--- Code Subgraph Complete ---\n")
    # Return updates to parent state
    formatted_artifact = (
        f"```python\n{child_output['raw_code']}\n```\n"
        f"Validation: {child_output['review_status']}"
    )
    return {"code_artifact": formatted_artifact}

def assemble_deliverable_node(state: ParentState) -> dict:
    print("\n[Supervisor] Assembling final client deliverable...")
    domain = state["target_domain"]
    if domain == "research":
        delivery = f"DELIVERABLE (RESEARCH):\n{state['research_report']}"
    else:
        delivery = f"DELIVERABLE (CODE IMPLEMENTATION):\n{state['code_artifact']}"
    return {"final_delivery": delivery}

def route_to_subgraph(state: ParentState) -> str:
    if state["target_domain"] == "research":
        return "research_agent"
    return "code_agent"

def build_parent_graph():
    builder = StateGraph(ParentState)
    
    builder.add_node("supervisor", supervisor_router_node)
    builder.add_node("research_agent", call_research_subgraph_node)
    builder.add_node("code_agent", call_code_subgraph_node)
    builder.add_node("assembler", assemble_deliverable_node)
    
    builder.add_edge(START, "supervisor")
    
    builder.add_conditional_edges(
        "supervisor",
        route_to_subgraph,
        {
            "research_agent": "research_agent",
            "code_agent": "code_agent"
        }
    )
    
    builder.add_edge("research_agent", "assembler")
    builder.add_edge("code_agent", "assembler")
    builder.add_edge("assembler", END)
    
    return builder.compile()


# =====================================================================
# 4. EXECUTION
# =====================================================================
def main():
    print("=" * 70)
    print("LangGraph: Hierarchical Subgraphs & Isolated State Communication")
    print("=" * 70)
    
    parent_app = build_parent_graph()
    
    # Test 1: Subgraph routing to Code Agent
    print("\n>>> TEST RUN 1: Code Request <<<")
    result_code = parent_app.invoke({
        "request": "Implement an asynchronous batch execution data pipeline in Python",
        "target_domain": "",
        "research_report": "",
        "code_artifact": "",
        "final_delivery": ""
    })
    print("\n" + "=" * 40)
    print(result_code["final_delivery"])
    print("=" * 40)
    
    # Test 2: Subgraph routing to Research Agent
    print("\n>>> TEST RUN 2: Research Request <<<")
    result_research = parent_app.invoke({
        "request": "Analyze state persistence guarantees across distributed agent swarms",
        "target_domain": "",
        "research_report": "",
        "code_artifact": "",
        "final_delivery": ""
    })
    print("\n" + "=" * 40)
    print(result_research["final_delivery"])
    print("=" * 40)

if __name__ == "__main__":
    main()
