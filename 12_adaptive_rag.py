import os
from typing import Annotated, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- 1. Define State ---
class GraphState(TypedDict):
    question: str
    generation: str
    source: str # 'web' or 'vectorstore'

# --- 2. Mock Tools / Services ---
def mock_web_search(question: str) -> str:
    print("---EXECUTING WEB SEARCH---")
    return f"Web search results for: {question}. Found latest info."

def mock_vector_store_retriever(question: str) -> str:
    print("---EXECUTING VECTOR STORE RETRIEVAL---")
    return f"Vector store internal docs for: {question}."

# --- 3. Route Output Schema ---
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

# --- 4. Nodes ---
def web_search_node(state: GraphState):
    question = state["question"]
    docs = mock_web_search(question)
    return {"generation": docs, "source": "web"}

def retrieve_node(state: GraphState):
    question = state["question"]
    docs = mock_vector_store_retriever(question)
    return {"generation": docs, "source": "vectorstore"}

# --- 5. Conditional Routing Logic ---
def route_question(state: GraphState):
    """
    Route question to web search or RAG.
    """
    print("---ROUTING QUESTION---")
    question = state["question"]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to 'agents', 'prompt engineering', and 'LLMs'.
    Use the vectorstore for questions on these topics. Otherwise, use web-search."""
    
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
    
    question_router = route_prompt | structured_llm_router
    source = question_router.invoke({"question": question})
    
    if source.datasource == 'web_search':
        print("---ROUTE: WEB SEARCH---")
        return "web_search"
    elif source.datasource == 'vectorstore':
        print("---ROUTE: VECTOR STORE---")
        return "vectorstore"

# --- 6. Graph Definition ---
workflow = StateGraph(GraphState)

workflow.add_node("web_search_node", web_search_node)
workflow.add_node("retrieve_node", retrieve_node)

# We use the conditional edge directly from the START node
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search_node",
        "vectorstore": "retrieve_node",
    }
)

workflow.add_edge("web_search_node", END)
workflow.add_edge("retrieve_node", END)

app = workflow.compile()

if __name__ == "__main__":
    print("=== Testing Adaptive RAG ===")
    
    # Test 1: Should route to vector store
    inputs1 = {"question": "What is an LLM agent?"}
    print(f"\nQuestion: {inputs1['question']}")
    for output in app.stream(inputs1):
        pass
    print("Source used:", output[list(output.keys())[0]]["source"])
    
    # Test 2: Should route to web search
    inputs2 = {"question": "What is the weather in Tokyo today?"}
    print(f"\nQuestion: {inputs2['question']}")
    for output in app.stream(inputs2):
        pass
    print("Source used:", output[list(output.keys())[0]]["source"])
