import operator
import sqlite3
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# --- State Definitions ---
class OverallState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    research_topics: List[str]
    summaries: Annotated[List[str], operator.add]
    draft: str
    next_agent: str

class SummarizeState(TypedDict):
    topic: str

# Structured output for topic generation
class Topics(BaseModel):
    topics: List[str] = Field(description="List of specific internet search queries (max 3)")

# --- Nodes ---
def supervisor(state: OverallState):
    print("Supervisor checking state...")
    if state.get("draft"):
        return {"next_agent": "FINISH"}
    if not state.get("research_topics"):
        return {"next_agent": "researcher"}
    return {"next_agent": "writer"}

def researcher(state: OverallState):
    print("Researcher is working...")
    messages = state.get("messages", [])
    user_request = messages[0].content if messages and hasattr(messages[0], 'content') else str(messages)
    
    prompt = f"Based on the user request, generate up to 3 specific internet search queries to gather comprehensive information.\nRequest: {user_request}"
    
    structured_llm = llm.with_structured_output(Topics)
    result = structured_llm.invoke(prompt)
    
    return {
        "research_topics": result.topics, 
        "messages": [AIMessage(content=f"Researcher identified topics: {', '.join(result.topics)}")]
    }

def research_worker(state: SummarizeState):
    print(f"Research Worker investigating: {state['topic']}")
    topic = state["topic"]
    
    # Use Tavily Search
    search = TavilySearchResults(max_results=2)
    try:
        docs = search.invoke(topic)
        docs_text = "\n".join([f"- {d['content']}" for d in docs])
    except Exception as e:
        docs_text = f"Failed to search: {e}"
        
    summary_prompt = f"Summarize the following search results for the topic '{topic}':\n{docs_text}"
    res = llm.invoke(summary_prompt)
    
    return {"summaries": [f"**{topic}**: {res.content}"]}

def map_research(state: OverallState):
    topics = state.get("research_topics", [])
    return [Send("research_worker", {"topic": t}) for t in topics]

def writer(state: OverallState):
    print("Writer is drafting...")
    summaries = state.get("summaries", [])
    summaries_text = "\n\n".join(summaries)
    
    messages = state.get("messages", [])
    user_request = messages[0].content if messages and hasattr(messages[0], 'content') else str(messages)
    
    prompt = f"Write a comprehensive, well-structured report based on these research summaries:\n{summaries_text}\n\nOriginal User request: {user_request}"
    res = llm.invoke(prompt)
    
    return {"draft": res.content, "messages": [AIMessage(content="Writer: Draft completed!")]}

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
    
    # Persistence
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # Human in the loop! Interrupt before writer drafts the report.
    return builder.compile(checkpointer=memory, interrupt_before=["writer"])

graph = build_graph()
