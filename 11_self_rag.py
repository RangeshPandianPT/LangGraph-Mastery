import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- 1. Define State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[str]

# --- 2. Mock Retriever (to keep tutorial simple without external DB) ---
def retrieve_documents(question: str) -> List[str]:
    """Mock retriever."""
    if "langchain" in question.lower():
        return [
            "LangChain is a framework for developing applications powered by language models.",
            "LangChain provides tools to connect LLMs to other sources of data."
        ]
    elif "langgraph" in question.lower():
        return [
            "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
            "LangGraph models workflows as graphs."
        ]
    return ["The sky is blue and the sun is bright."]

# --- 3. Structured Outputs for Grading ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# --- 4. Define Nodes ---
def retrieve(state: GraphState):
    """Node to retrieve documents"""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retrieve_documents(question)
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    """Node to filter out irrelevant documents"""
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(GradeDocuments)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question. "
                   "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. "
                   "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    
    chain = prompt | structured_llm
    
    filtered_docs = []
    for d in documents:
        score = chain.invoke({"question": question, "document": d})
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT IRRELEVANT---")
            
    return {"documents": filtered_docs, "question": question}

def generate(state: GraphState):
    """Node to generate answer"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n\nContext: {context}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    
    context = "\n".join(documents)
    response = chain.invoke({"context": context, "question": question})
    
    return {"generation": response.content, "documents": documents, "question": question}

def transform_query(state: GraphState):
    """Node to rewrite the query if documents are irrelevant"""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query rewrite optimizer. Look at the input question and try to reason about the underlying semantic intent / meaning. Write a better, more specific question."),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": question})
    
    return {"documents": documents, "question": response.content}

# --- 5. Define Conditional Edges ---
def decide_to_generate(state: GraphState):
    """Decide whether to generate or rewrite query based on filtered documents."""
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        # All documents were deemed irrelevant
        print("---DECISION: ALL DOCUMENTS ARE IRRELEVANT, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents
        print("---DECISION: GENERATE---")
        return "generate"

# --- 6. Build the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()

if __name__ == "__main__":
    print("=== Testing Self-RAG ===")
    inputs = {"question": "What is LangGraph?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished node: {key}")
            
    print("\nFinal Generation:")
    print(value.get("generation"))
