from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
import chromadb
from neo4j import GraphDatabase
import requests
import json
import os

# Configuration
VECTOR_DB = os.path.join(os.path.dirname(__file__), "..", "vector_db")
COLLECTION_NAME = "resume_collection"
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password123"  # Updated to match neo4j_import.py
NEO4J_DATABASE = "neo4j"
OLLAMA_URL = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen2.5:7b"

# State Definition
class AgentState(TypedDict):
    question: str
    resume_ids: list
    vector_context: list
    graph_context: list
    answer: str
    error: str

# Nodes
def retrieve_vector(state: AgentState) -> AgentState:
    """Retrieves relevant resume IDs and text from ChromaDB based on the question."""
    print("Node: retrieve_vector")
    try:
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        
        results = collection.query(
            query_texts=[state["question"]],
            n_results=3
        )
        
        resume_ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        
        return {
            "resume_ids": resume_ids,
            "vector_context": documents,
            "error": ""
        }
    except Exception as e:
        return {"error": f"Vector retrieval failed: {str(e)}"}

def retrieve_graph(state: AgentState) -> AgentState:
    """Retrieves structured relationships from Neo4j for the identified resumes."""
    print("Node: retrieve_graph")
    resume_ids = state.get("resume_ids", [])
    if not resume_ids:
        return {"graph_context": []}
        
    graph_context = []
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        query = """
        MATCH (r:Resume)
        WHERE r.resume_id IN $resume_ids
        OPTIONAL MATCH (r)-[rel]-(n)
        RETURN r.resume_id AS resume_id, 
               properties(r) AS properties,
               collect({
                   relationship: type(rel),
                   node_labels: labels(n),
                   node_properties: properties(n)
               }) AS connections
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, resume_ids=resume_ids)
            for record in result:
                graph_context.append({
                    "resume_id": record["resume_id"],
                    "properties": record["properties"],
                    "connections": record["connections"]
                })
        driver.close()
        return {"graph_context": graph_context, "error": ""}
    except Exception as e:
        return {"error": f"Graph retrieval failed: {str(e)}"}

def generate_answer(state: AgentState) -> AgentState:
    """Uses Ollama to generate an answer combining both contexts."""
    print("Node: generate_answer")
    question = state.get("question")
    vector_context = state.get("vector_context", [])
    graph_context = state.get("graph_context", [])
    error = state.get("error", "")
    
    if error:
        return {"answer": f"I encountered an error during retrieval: {error}"}
        
    context = json.dumps({
        "vector_data": vector_context,
        "graph_data": graph_context
    }, indent=2, default=str)
    
    prompt = f"""
You are an expert HR and recruitment assistant.
Answer the user's question using ONLY the provided context retrieved from our hybrid database.

USER QUESTION:
{question}

RETRIEVED CONTEXT (Vector & Graph Data):
{context}

Instructions:
1. Identify relevant candidates by Name and Resume ID.
2. Synthesize their skills, experience, and projects.
3. If the context does not contain the answer, politely state that you cannot find the information.
4. Keep the answer professional and well-structured.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": QWEN_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return {"answer": data["response"]}
    except Exception as e:
        return {"answer": f"Generation failed: {str(e)}"}

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve_vector", retrieve_vector)
workflow.add_node("retrieve_graph", retrieve_graph)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("retrieve_vector")
workflow.add_edge("retrieve_vector", "retrieve_graph")
workflow.add_edge("retrieve_graph", "generate_answer")
workflow.add_edge("generate_answer", END)

app_graph = workflow.compile()

def run_agent(question: str):
    """Utility function to run the agent from other scripts."""
    inputs = {"question": question, "resume_ids": [], "vector_context": [], "graph_context": [], "answer": "", "error": ""}
    result = app_graph.invoke(inputs)
    return result

if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "Who has experience with Python?"
    res = run_agent(question)
    print("\n--- FINAL ANSWER ---")
    print(res["answer"])
