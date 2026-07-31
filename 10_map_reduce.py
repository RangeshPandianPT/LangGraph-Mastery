import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

class OverallState(TypedDict):
    documents: List[str]
    summaries: Annotated[List[str], operator.add]

class SummarizeState(TypedDict):
    document: str

def generate_docs(state: OverallState):
    print("Generating documents...")
    return {"documents": ["Doc 1: AI is great", "Doc 2: LangGraph is powerful", "Doc 3: Python is fun"]}

def summarize_doc(state: SummarizeState):
    # This node will run in parallel for each document
    doc = state["document"]
    print(f"Summarizing: {doc}")
    summary = f"Summary of [{doc}]"
    # We return an update matching the OverallState's reducer key
    return {"summaries": [summary]}

def map_documents(state: OverallState):
    # This is the conditional edge function that dynamically creates Send objects
    # It fans out to the "summarize_doc" node for each document
    docs = state.get("documents", [])
    print(f"Fanning out to {len(docs)} summary tasks...")
    
    # Return a list of Send operations
    return [Send("summarize_doc", {"document": d}) for d in docs]

def main():
    builder = StateGraph(OverallState)
    builder.add_node("generate_docs", generate_docs)
    builder.add_node("summarize_doc", summarize_doc)
    
    builder.add_edge(START, "generate_docs")
    
    # Add a conditional edge from generate_docs to summarize_doc
    # The map_documents function will return the Send commands
    builder.add_conditional_edges("generate_docs", map_documents, ["summarize_doc"])
    
    # After parallel summarization, go to END
    builder.add_edge("summarize_doc", END)
    
    graph = builder.compile()
    
    print("--- Map-Reduce Workflow Starting ---")
    result = graph.invoke({"documents": [], "summaries": []})
    
    print("\n--- Final Summaries ---")
    for s in result["summaries"]:
        print(s)

if __name__ == "__main__":
    main()
