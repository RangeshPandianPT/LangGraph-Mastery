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
    evaluation: str
    revision_count: int
    next_agent: str

class SummarizeState(TypedDict):
    topic: str

# Structured output for topic generation
class Topics(BaseModel):
    topics: List[str] = Field(description="List of specific internet search queries (max 3)")
    
class Evaluation(BaseModel):
    is_acceptable: bool = Field(description="Whether the draft fully answers the user's request")
    feedback: str = Field(description="Feedback on what is missing or needs improvement")

# --- Nodes ---
def supervisor(state: OverallState):
    print("Supervisor checking state...")
    if state.get("evaluation") == "ACCEPT":
        return {"next_agent": "FINISH"}
    if state.get("draft") and state.get("evaluation") != "REJECT":
        return {"next_agent": "evaluator"}
    if not state.get("research_topics"):
        return {"next_agent": "researcher"}
    return {"next_agent": "writer"}

def researcher(state: OverallState):
    print("Researcher is working...")
    messages = state.get("messages", [])
    user_request = messages[0].content if messages and hasattr(messages[0], 'content') else str(messages)
    
    # If there is feedback, incorporate it
    feedback = state.get("evaluation", "")
    if feedback and feedback != "ACCEPT":
        prompt = f"Based on the user request and feedback on a previous draft, generate up to 3 NEW specific internet search queries to address the missing information.\nRequest: {user_request}\nFeedback: {feedback}"
    else:
        prompt = f"Based on the user request, generate up to 3 specific internet search queries to gather comprehensive information.\nRequest: {user_request}"
    
    structured_llm = llm.with_structured_output(Topics)
    result = structured_llm.invoke(prompt)
    
    return {
        "research_topics": result.topics, 
        "evaluation": "", # Clear evaluation so supervisor routes correctly
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
    feedback = state.get("evaluation", "")
    
    prompt = f"Write a comprehensive, well-structured report based on these research summaries:\n{summaries_text}\n\nOriginal User request: {user_request}"
    if feedback and feedback != "ACCEPT":
        prompt += f"\n\nImprove your previous draft based on this feedback from the Evaluator: {feedback}"
        
    res = llm.invoke(prompt)
    
    # Reset research topics so a new set can be generated if rejected
    return {"draft": res.content, "messages": [AIMessage(content="Writer: Draft completed!")], "research_topics": []}

def evaluator(state: OverallState):
    print("Evaluator is reviewing...")
    messages = state.get("messages", [])
    user_request = messages[0].content if messages and hasattr(messages[0], 'content') else str(messages)
    draft = state.get("draft", "")
    revision_count = state.get("revision_count", 0)
    
    if revision_count >= 2:
        # Max revisions reached
        return {"evaluation": "ACCEPT", "revision_count": revision_count + 1, "messages": [AIMessage(content="Evaluator: Max revisions reached, accepting draft as final.")]}
        
    prompt = f"Evaluate this draft against the original user request.\nRequest: {user_request}\nDraft: {draft}\nDoes it fully and accurately address the request?"
    
    structured_llm = llm.with_structured_output(Evaluation)
    result = structured_llm.invoke(prompt)
    
    if result.is_acceptable:
        return {"evaluation": "ACCEPT", "revision_count": revision_count + 1, "messages": [AIMessage(content="Evaluator: Draft approved!")]}
    else:
        return {"evaluation": result.feedback, "revision_count": revision_count + 1, "messages": [AIMessage(content=f"Evaluator: Revision needed. Feedback: {result.feedback}")]}

# --- Routing ---
def supervisor_router(state: OverallState):
    if state["next_agent"] == "FINISH":
        return END
    if state["next_agent"] == "evaluator":
        return "evaluator"
    if state["next_agent"] == "researcher":
        return "researcher"
    return "writer"

def build_graph():
    builder = StateGraph(OverallState)
    
    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher", researcher)
    builder.add_node("research_worker", research_worker)
    builder.add_node("writer", writer)
    builder.add_node("evaluator", evaluator)
    
    builder.add_edge(START, "supervisor")
    
    builder.add_conditional_edges(
        "supervisor", 
        supervisor_router, 
        {"researcher": "researcher", "writer": "writer", "evaluator": "evaluator", END: END}
    )
    
    # Researcher fans out to research_workers (Map)
    builder.add_conditional_edges("researcher", map_research, ["research_worker"])
    
    # Workers fan in to supervisor
    builder.add_edge("research_worker", "supervisor")
    
    # Writer goes to supervisor
    builder.add_edge("writer", "supervisor")
    
    # Evaluator goes to supervisor
    builder.add_edge("evaluator", "supervisor")
    
    # Persistence
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # Human in the loop! Interrupt before writer drafts the report.
    return builder.compile(checkpointer=memory, interrupt_before=["writer"])

graph = build_graph()
