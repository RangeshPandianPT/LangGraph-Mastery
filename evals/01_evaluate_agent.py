import os
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict

# --- 1. Define a Simple Graph to Evaluate ---
class State(TypedDict):
    question: str
    answer: str

def generate_answer(state: State):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer concisely."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"question": state["question"]})
    return {"answer": response.content}

workflow = StateGraph(State)
workflow.add_node("generate", generate_answer)
workflow.set_entry_point("generate")
workflow.add_edge("generate", END)
app = workflow.compile()

# --- 2. Evaluation Setup ---
def predict(inputs: dict) -> dict:
    """Wrapper function that LangSmith evaluation will call."""
    # We invoke the graph with the given inputs
    result = app.invoke(inputs)
    # The output format must match what the evaluators expect
    return {"output": result["answer"]}

def run_evaluation():
    # Note: Requires LANGCHAIN_API_KEY to be set in environment
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("Warning: LANGCHAIN_API_KEY not set. Cannot run LangSmith evaluation.")
        return
        
    client = Client()
    
    # 1. Create a Dataset programmatically (or you can do this in the UI)
    dataset_name = "LangGraph-QA-Eval-Dataset"
    
    try:
        # Check if dataset exists, if not create it
        client.read_dataset(dataset_name=dataset_name)
        print(f"Dataset '{dataset_name}' already exists.")
    except Exception:
        print(f"Creating dataset '{dataset_name}'...")
        dataset = client.create_dataset(dataset_name, description="QA dataset for LangGraph evaluation")
        
        # Add examples
        examples = [
            ("What is the capital of France?", "Paris"),
            ("Who wrote Hamlet?", "William Shakespeare"),
            ("What is 2 + 2?", "4")
        ]
        
        for q, a in examples:
            client.create_example(
                inputs={"question": q},
                outputs={"expected": a},
                dataset_id=dataset.id,
            )
            
    # 2. Define Evaluators
    # We use a standard correctness evaluator using an LLM to judge the output
    qa_evaluator = LangChainStringEvaluator("qa")
    
    # 3. Run the evaluation
    print(f"Running evaluation on dataset '{dataset_name}'...")
    experiment_results = evaluate(
        predict, # The function that runs our graph
        data=dataset_name,
        evaluators=[qa_evaluator],
        experiment_prefix="LangGraph-Agent-Eval",
        metadata={"version": "1.0", "agent_type": "simple_qa"}
    )
    
    print("\nEvaluation complete! View the results in the LangSmith UI.")

if __name__ == "__main__":
    print("=== LangGraph Agent Evaluation ===")
    run_evaluation()
