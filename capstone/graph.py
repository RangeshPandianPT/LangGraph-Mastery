import operator
import sqlite3
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.sqlite import SqliteSaver
import time

# --- State Definitions ---
class OverallState(TypedDict):
    messages: Annotated[List[str], operator.add]
    research_topics: List[str]
    summaries: Annotated[List[str], operator.add]
    draft: str
    next_agent: str

class SummarizeState(TypedDict):
    topic: str

# --- Nodes ---
def supervisor(state: OverallState):
    print("Supervisor checking state...")
    messages = state.get("messages", [])
    if state.get("draft"):
        return {"next_agent": "FINISH"}
    if not state.get("research_topics"):
        return {"next_agent": "researcher"}
    return {"next_agent": "writer"}

def researcher(state: OverallState):
    print("Researcher is working...")
    last_message = state["messages"][-1] if state.get("messages") else "AI"
    # Extract fake topics based on the prompt
    topics = [f"Topic 1 for {last_message}", f"Topic 2 for {last_message}"]
    return {"research_topics": topics, "messages": ["Researcher: Found topics to investigate."]}

def research_worker(state: SummarizeState):
    # This node simulates a Map-Reduce parallel task
    time.sleep(1) # Simulate API call/Streaming
    topic = state["topic"]
    return {"summaries": [f"Deep dive into {topic}"]}

def map_research(state: OverallState):
    # Conditional edge to fan out
    topics = state.get("research_topics", [])
    return [Send("research_worker", {"topic": t}) for t in topics]

def writer(state: OverallState):
    print("Writer is drafting...")
    time.sleep(1)
    summaries = state.get("summaries", [])
    draft = f"Here is the final report based on: {', '.join(summaries)}"
    return {"draft": draft, "messages": ["Writer: Draft completed!"]}

# --- Routing ---
def supervisor_router(state: OverallState):
    if state["next_agent"] == "FINISH":
        return END
    if state["next_agent"] == "researcher":
        return "researcher"
    return "writer"

def build_graph():
    builder = StateGraph(OverallState)
    
    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher", researcher)
    builder.add_node("research_worker", research_worker)
    builder.add_node("writer", writer)
    
    # Workflow
    builder.add_edge(START, "supervisor")
    
    builder.add_conditional_edges(
        "supervisor", 
        supervisor_router, 
        {"researcher": "researcher", "writer": "writer", END: END}
    )
    
    # Researcher fans out to research_workers (Map)
    builder.add_conditional_edges("researcher", map_research, ["research_worker"])
    
    # Workers fan in to supervisor
    builder.add_edge("research_worker", "supervisor")
    
    # Writer goes to supervisor
    builder.add_edge("writer", "supervisor")
    
    # Add Persistence (SQLite) for Memory and Time Travel
    conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    return builder.compile(checkpointer=memory)

graph = build_graph()
