"""
19_corrective_rag.py: Corrective RAG (CRAG) with Knowledge Refinement & Fallback

What you will learn:
1. The limitations of naive RAG and Self-RAG.
2. The CRAG workflow (Corrective Retrieval-Augmented Generation):
   - Document Grading: Evaluator grades documents as 'CORRECT', 'AMBIGUOUS', or 'INCORRECT'.
   - Knowledge Refinement: Decomposes documents into knowledge strips and filters out noise.
   - Web Search Fallback: Automatically triggered if local documents are ambiguous or incorrect.
3. How to complete the Advanced RAG Trilogy (Self-RAG -> Adaptive-RAG -> Corrective-RAG).
"""

import os
from typing import TypedDict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# --- 1. Graph State ---
class CRAGState(TypedDict):
    question: str
    documents: List[str]
    refined_knowledge: List[str]
    web_results: List[str]
    retrieval_status: str  # 'CORRECT', 'AMBIGUOUS', 'INCORRECT'
    generation: str

# --- 2. Mock Knowledge Base (Local Documents) ---
LOCAL_KB = {
    "langgraph checkpointer": [
        "LangGraph checkpointers save thread state snapshots at each superstep. This enables persistence and time travel.",
        "PostgresSaver and MemorySaver are standard checkpointer implementations in LangGraph."
    ],
    "quantum computing": [
        "Quantum computing utilizes qubits that can exist in superposition. Superconducting circuits are a common physical implementation."
    ]
}

def mock_vector_search(query: str) -> List[str]:
    q_lower = query.lower()
    for topic, docs in LOCAL_KB.items():
        if any(w in q_lower for w in topic.split()):
            return docs
    # Irrelevant or empty match
    return ["The weather in Zurich is cool and cloudy today. The local train network is punctual."]

def mock_web_search(query: str) -> List[str]:
    print(f"  [Web Search] Querying web search index for: '{query}'...")
    return [
        f"Web Snippet: Authoritative web documentation on '{query}'. Released in late 2024 with enhanced production features.",
        f"Web Snippet: Community benchmark showing high reliability and active developer adoption for '{query}'."
    ]

# --- 3. Structured Output for Document Evaluation ---
class GradeResult(BaseModel):
    grade: str = Field(description="Must be 'CORRECT', 'AMBIGUOUS', or 'INCORRECT'")
    rationale: str = Field(description="Brief reason for the grade")

# --- 4. Node Definitions ---

def retrieve_node(state: CRAGState) -> dict:
    """Retrieves initial documents from the local knowledge base."""
    question = state["question"]
    print(f"\n[RETRIEVE] Fetching local documents for question: '{question}'")
    docs = mock_vector_search(question)
    print(f"Retrieved {len(docs)} documents.")
    return {"documents": docs}

def grade_documents_node(state: CRAGState) -> dict:
    """Evaluates the retrieved documents against the question."""
    question = state["question"]
    docs = state["documents"]
    print("\n[GRADE DOCUMENTS] Evaluating document relevance and confidence...")
    
    combined_docs = " ".join(docs)
    
    # LLM evaluation if API key is provided
    if os.getenv("OPENAI_API_KEY"):
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            structured_llm = llm.with_structured_output(GradeResult)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an evaluator assessing whether retrieved documents can answer a user question. "
                           "Grade as 'CORRECT' if docs clearly answer the question, 'AMBIGUOUS' if partially relevant, or 'INCORRECT' if irrelevant."),
                ("human", "Question: {question}\n\nDocuments:\n{docs}")
            ])
            result = (prompt | structured_llm).invoke({"question": question, "docs": combined_docs})
            status = result.grade.upper()
            print(f"LLM Evaluator status: {status} ({result.rationale})")
            return {"retrieval_status": status}
        except Exception as e:
            print(f"[NOTE] LLM grade fallback: {e}")
            
    # Deterministic heuristic fallback for demonstration
    keywords = [w.lower() for w in question.split() if len(w) > 3]
    matches = sum(1 for kw in keywords if kw in combined_docs.lower())
    
    if matches >= 2:
        status = "CORRECT"
    elif matches == 1:
        status = "AMBIGUOUS"
    else:
        status = "INCORRECT"
        
    print(f"Evaluator Score -> Status: {status} (Keyword overlaps: {matches})")
    return {"retrieval_status": status}

def refine_knowledge_node(state: CRAGState) -> dict:
    """Extracts granular knowledge strips, filtering out sentence noise."""
    print("\n[REFINE KNOWLEDGE] Decomposing chunks into filtered knowledge strips...")
    docs = state["documents"]
    question = state["question"]
    
    strips = []
    for doc in docs:
        sentences = doc.split(". ")
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Keep sentences that relate to query keywords
            if any(k in sentence.lower() for k in question.lower().split()):
                strips.append(sentence)
                
    if not strips and docs:
        strips = [docs[0]]  # Fallback to first chunk if strict filter yielded 0
        
    print(f"Refined {len(docs)} raw docs into {len(strips)} high-confidence knowledge strips:")
    for s in strips:
        print(f"   * {s}")
        
    return {"refined_knowledge": strips}

def web_search_fallback_node(state: CRAGState) -> dict:
    """Performs query transformation and searches web to correct knowledge deficit."""
    question = state["question"]
    status = state["retrieval_status"]
    print(f"\n[WEB SEARCH FALLBACK] Triggered due to '{status}' retrieval status.")
    
    # Query transformation
    search_query = f"{question} documentation technical guide"
    results = mock_web_search(search_query)
    return {"web_results": results}

def generate_answer_node(state: CRAGState) -> dict:
    """Synthesizes final answer from refined local knowledge and web results."""
    print("\n[GENERATE] Synthesizing final grounded response...")
    question = state["question"]
    refined = state.get("refined_knowledge", [])
    web = state.get("web_results", [])
    status = state.get("retrieval_status", "")
    
    context_sections = []
    if refined:
        context_sections.append("LOCAL REFINED KNOWLEDGE:\n" + "\n".join(f"- {r}" for r in refined))
    if web:
        context_sections.append("WEB SEARCH RESULTS:\n" + "\n".join(f"- {w}" for w in web))
        
    final_context = "\n\n".join(context_sections)
    
    answer = (
        f"Answer to: '{question}'\n"
        f"Retrieval Diagnostic: [{status}]\n\n"
        f"Synthesized Findings:\n"
        f"{final_context}\n\n"
        f"Conclusion: Accurate answer formulated with CRAG confidence safeguards."
    )
    return {"generation": answer}

# --- 5. Conditional Routing Logic ---
def route_after_grading(state: CRAGState) -> str:
    status = state["retrieval_status"]
    if status == "CORRECT":
        return "refine_knowledge"
    elif status == "AMBIGUOUS":
        # In ambiguous cases, we refine what we have AND search the web
        return "refine_and_search"
    else:  # INCORRECT
        return "web_search"

# --- 6. Build the Graph ---
def build_crag_graph():
    builder = StateGraph(CRAGState)
    
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("refine_knowledge", refine_knowledge_node)
    builder.add_node("web_search", web_search_fallback_node)
    builder.add_node("generate", generate_answer_node)
    
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    
    builder.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "refine_knowledge": "refine_knowledge",
            "refine_and_search": "refine_knowledge",
            "web_search": "web_search"
        }
    )
    
    # From refine_knowledge, if ambiguous also run web_search, else generate
    def route_after_refine(state: CRAGState) -> str:
        if state["retrieval_status"] == "AMBIGUOUS":
            return "web_search"
        return "generate"
        
    builder.add_conditional_edges(
        "refine_knowledge",
        route_after_refine,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )
    
    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", END)
    
    return builder.compile()

# --- 7. Main Execution ---
def main():
    print("=" * 70)
    print("LangGraph: Corrective RAG (CRAG) with Knowledge Refinement & Fallback")
    print("=" * 70)
    
    crag_app = build_crag_graph()
    
    # -------------------------------------------------------------
    # Scenario 1: High confidence local match (CORRECT)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> SCENARIO 1: High Confidence Query (Matches Local KB) <<<")
    print("=" * 50)
    res_correct = crag_app.invoke({
        "question": "How do LangGraph checkpointers save thread state snapshots?",
        "documents": [],
        "refined_knowledge": [],
        "web_results": [],
        "retrieval_status": "",
        "generation": ""
    })
    print("\n" + res_correct["generation"])
    
    # -------------------------------------------------------------
    # Scenario 2: Unmatched / Unknown Query (Triggers Web Fallback)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> SCENARIO 2: Unknown Query (Triggers Web Search Fallback) <<<")
    print("=" * 50)
    res_fallback = crag_app.invoke({
        "question": "What are the latest features in LangGraph v1.2 release?",
        "documents": [],
        "refined_knowledge": [],
        "web_results": [],
        "retrieval_status": "",
        "generation": ""
    })
    print("\n" + res_fallback["generation"])

if __name__ == "__main__":
    main()
