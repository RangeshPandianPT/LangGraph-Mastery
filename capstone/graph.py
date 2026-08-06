import operator
import sqlite3
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# New Imports for RAG and Code Execution
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.utilities import PythonREPL

load_dotenv()

# Initialize Ollama LLM
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(model="llama3.2:3b", temperature=0.2, base_url=ollama_base_url)

# --- State Definitions ---
class OverallState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    research_topics: List[str]
    summaries: Annotated[List[str], operator.add]
    draft: str
    fact_check_result: str
    evaluation: str
    revision_count: int
    next_agent: str

class SummarizeState(TypedDict):
    topic: str
    request: str

# Structured output for topic generation
class Topics(BaseModel):
    topics: List[str] = Field(description="List of specific internet search queries (max 3)")
    
class Evaluation(BaseModel):
    is_acceptable: bool = Field(description="Whether the draft fully answers the user's request")
    feedback: str = Field(description="Feedback on what is missing or needs improvement")

class FactCheck(BaseModel):
    is_accurate: bool = Field(description="Whether the draft accurately reflects the research summaries without hallucinating")
    feedback: str = Field(description="Feedback on what facts are inaccurate or missing")

# --- Nodes ---
def supervisor(state: OverallState):
    print("Supervisor checking state...")
    if state.get("evaluation") == "ACCEPT":
        return {"next_agent": "FINISH"}
    if state.get("draft") and state.get("fact_check_result") != "ACCEPT" and not (state.get("fact_check_result") and state.get("fact_check_result") != "ACCEPT"):
        return {"next_agent": "fact_checker"}
    if state.get("draft") and state.get("fact_check_result") == "ACCEPT" and state.get("evaluation") != "REJECT":
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
    
    # Use Tavily Search and Wikipedia
    tavily_search = TavilySearchResults(max_results=2)
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    docs_text = ""
    try:
        docs = tavily_search.invoke(topic)
        for d in docs:
            docs_text += f"- Title/Snippet: {d.get('content', '')}\n"
            url = d.get("url")
            if url:
                try:
                    loader = WebBaseLoader(url)
                    web_docs = loader.load()
                    if web_docs:
                        page_content = web_docs[0].page_content.replace('\n', ' ').strip()
                        docs_text += f"  Scraped Content: {page_content[:1000]}...\n"
                except Exception as scrape_err:
                    pass
    except Exception as e:
        docs_text += f"Failed to search web: {e}\n"
        
    try:
        wiki_docs = wikipedia.invoke(topic)
        docs_text += f"\n- Wikipedia Summary:\n{wiki_docs}\n"
    except Exception as e:
        docs_text += f"Failed to search Wikipedia: {e}\n"
        
    summary_prompt = f"Summarize the following search results and scraped content for the topic '{topic}':\n{docs_text}"
    res = llm.invoke(summary_prompt)
    
    return {"summaries": [f"**Web Research - {topic}**: {res.content}"]}

def document_retriever(state: SummarizeState):
    print("Document Retriever checking local DB...")
    req = state["request"]
    try:
        embeddings = OllamaEmbeddings(model="llama3.2:3b", base_url=ollama_base_url)
        vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
        docs = vectorstore.similarity_search(req, k=2)
        if docs:
            doc_text = "\n\n".join([d.page_content for d in docs])
            return {"summaries": [f"**Local Documents**: {doc_text}"]}
    except Exception as e:
        print("Vectorstore error:", e)
    return {"summaries": ["**Local Documents**: No relevant local documents found."]}

def data_analyst(state: SummarizeState):
    print("Data Analyst exploring data...")
    req = state["request"]
    prompt = f"Based on this request: '{req}', write Python code to perform any relevant math, logic, or data calculations. Output ONLY valid executable python code. No markdown formatting. If no calculation is needed, output: print('No analysis needed')\n\nCRITICAL SECURITY RULE: Do not import or use 'os', 'sys', 'subprocess', or execute any system commands."
    res = llm.invoke(prompt)
    code = res.content.replace('```python', '').replace('```', '').strip()
    
    # Basic Security Check
    forbidden_imports = ["import os", "import sys", "import subprocess", "__import__", "eval", "exec"]
    if any(forbidden in code for forbidden in forbidden_imports):
        return {"summaries": ["**Data Analyst Error**: Security violation detected. Code execution blocked."]}
        
    try:
        repl = PythonREPL()
        result = repl.run(code)
        return {"summaries": [f"**Data Analyst Output**: {result.strip()}"]}
    except Exception as e:
        return {"summaries": [f"**Data Analyst Error**: {e}"]}

def map_research(state: OverallState):
    topics = state.get("research_topics", [])
    messages = state.get("messages", [])
    req = messages[0].content if messages and hasattr(messages[0], 'content') else str(messages)
    
    sends = [Send("research_worker", {"topic": t, "request": req}) for t in topics]
    sends.append(Send("document_retriever", {"topic": "Local Documents", "request": req}))
    sends.append(Send("data_analyst", {"topic": "Data Analysis", "request": req}))
    return sends

def writer(state: OverallState):
    print("Writer is drafting...")
    summaries = state.get("summaries", [])
    summaries_text = "\n\n".join(summaries)
    
    messages = state.get("messages", [])
    user_request = messages[0].content if messages and hasattr(messages[0], 'content') else str(messages)
    feedback = state.get("evaluation", "")
    fact_feedback = state.get("fact_check_result", "")
    
    prompt = f"Write a comprehensive, well-structured report based on these research summaries (which include Web Search, Local Documents, and Data Analysis):\n{summaries_text}\n\nOriginal User request: {user_request}"
    if feedback and feedback != "ACCEPT":
        prompt += f"\n\nImprove your previous draft based on this feedback from the Evaluator: {feedback}"
    if fact_feedback and fact_feedback != "ACCEPT":
        prompt += f"\n\nFix the following factual inaccuracies in your draft based on the Fact Checker's feedback: {fact_feedback}"
        
    res = llm.invoke(prompt)
    
    return {"draft": res.content, "messages": [AIMessage(content="Writer: Draft completed!")], "research_topics": [], "fact_check_result": "", "evaluation": ""}

def fact_checker(state: OverallState):
    print("Fact Checker is verifying...")
    draft = state.get("draft", "")
    summaries = "\n\n".join(state.get("summaries", []))
    
    prompt = f"Verify the following draft against the provided research summaries. \n\nSummaries:\n{summaries}\n\nDraft:\n{draft}\n\nDoes the draft accurately reflect the research without hallucinating information?"
    
    structured_llm = llm.with_structured_output(FactCheck)
    result = structured_llm.invoke(prompt)
    
    if result.is_accurate:
        return {"fact_check_result": "ACCEPT", "messages": [AIMessage(content="Fact Checker: Draft is factually accurate.")]}
    else:
        return {"fact_check_result": result.feedback, "messages": [AIMessage(content=f"Fact Checker: Inaccuracies found. {result.feedback}")]}

def evaluator(state: OverallState):
    print("Evaluator is reviewing...")
    messages = state.get("messages", [])
    user_request = messages[0].content if messages and hasattr(messages[0], 'content') else str(messages)
    draft = state.get("draft", "")
    revision_count = state.get("revision_count", 0)
    
    if revision_count >= 2:
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
    if state["next_agent"] == "fact_checker":
        return "fact_checker"
    if state["next_agent"] == "researcher":
        return "researcher"
    return "writer"

def build_graph():
    builder = StateGraph(OverallState)
    
    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher", researcher)
    builder.add_node("research_worker", research_worker)
    builder.add_node("document_retriever", document_retriever)
    builder.add_node("data_analyst", data_analyst)
    builder.add_node("writer", writer)
    builder.add_node("fact_checker", fact_checker)
    builder.add_node("evaluator", evaluator)
    
    builder.add_edge(START, "supervisor")
    
    builder.add_conditional_edges(
        "supervisor", 
        supervisor_router, 
        {"researcher": "researcher", "writer": "writer", "fact_checker": "fact_checker", "evaluator": "evaluator", END: END}
    )
    
    # Researcher fans out to research_workers, document_retriever, data_analyst (Map)
    builder.add_conditional_edges("researcher", map_research, ["research_worker", "document_retriever", "data_analyst"])
    
    # Workers fan in to supervisor
    builder.add_edge("research_worker", "supervisor")
    builder.add_edge("document_retriever", "supervisor")
    builder.add_edge("data_analyst", "supervisor")
    
    # Writer goes to supervisor
    builder.add_edge("writer", "supervisor")
    
    # Fact Checker goes to supervisor
    builder.add_edge("fact_checker", "supervisor")
    
    # Evaluator goes to supervisor
    builder.add_edge("evaluator", "supervisor")
    
    # Persistence
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    return builder.compile(checkpointer=memory, interrupt_before=["writer"])

graph = build_graph()
