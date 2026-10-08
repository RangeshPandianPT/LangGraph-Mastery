import os
import sys
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI

# Add parent directory to path to import our hybrid rag agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib

# Dynamically import the script to avoid running its main block
hybrid_rag = importlib.import_module("14_hybrid_rag_agent")

def predict(inputs: dict) -> dict:
    """Wrapper function that LangSmith evaluation will call to test our RAG graph."""
    result = hybrid_rag.app_graph.invoke({
        "question": inputs["question"], 
        "resume_ids": [], 
        "vector_context": [], 
        "graph_context": [], 
        "answer": "", 
        "error": ""
    })
    
    return {
        "output": result.get("answer", ""),
        "retrieved_vector": str(result.get("vector_context", [])),
        "retrieved_graph": str(result.get("graph_context", []))
    }

def rag_groundedness_evaluator(run: Run, example: Example) -> dict:
    """Evaluates if the answer is grounded in the retrieved Hybrid context."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    answer = run.outputs["output"]
    vector_context = run.outputs["retrieved_vector"]
    graph_context = run.outputs["retrieved_graph"]
    
    prompt = f"""
Evaluate whether the Assistant's answer is supported by the retrieved context.
If the answer is supported by the context, reply 'YES'. If it contains hallucinated information not in the context, reply 'NO'.
If the assistant correctly states that the information is not in the context, reply 'YES'.

Answer: {answer}
Vector Context: {vector_context}
Graph Context: {graph_context}

Output only YES or NO.
"""
    response = llm.invoke(prompt)
    score = 1 if "YES" in response.content.upper() else 0
    return {"key": "groundedness", "score": score}

def run_evaluation():
    # Note: Requires LANGCHAIN_API_KEY and OPENAI_API_KEY to be set in environment
    if not os.getenv("LANGCHAIN_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("Warning: LANGCHAIN_API_KEY and OPENAI_API_KEY must be set. Cannot run LangSmith evaluation.")
        return
        
    client = Client()
    
    dataset_name = "Hybrid-RAG-Eval-Dataset"
    
    try:
        # Check if dataset exists, if not create it
        client.read_dataset(dataset_name=dataset_name)
        print(f"Dataset '{dataset_name}' already exists.")
    except Exception:
        print(f"Creating dataset '{dataset_name}'...")
        dataset = client.create_dataset(dataset_name, description="QA dataset for Hybrid RAG evaluation")
        
        # Add basic examples to test our graph
        examples = [
            ("Who has experience with Python?", "Candidate should be listed with Python experience."),
            ("What are the skills of the candidate who worked as a Data Scientist?", "Skills related to Data Science should be listed."),
            ("Tell me about a candidate who does not exist.", "Assistant should state the information is not found.")
        ]
        
        for q, expected in examples:
            client.create_example(
                inputs={"question": q},
                outputs={"expected": expected},
                dataset_id=dataset.id,
            )
            
    print(f"Running evaluation on dataset '{dataset_name}'...")
    # Evaluate our Hybrid RAG Agent using the custom Groundedness Evaluator
    experiment_results = evaluate(
        predict, 
        data=dataset_name,
        evaluators=[rag_groundedness_evaluator], 
        experiment_prefix="Hybrid-RAG-Eval",
        metadata={"version": "1.0", "agent_type": "hybrid_rag"}
    )
    
    print("\nEvaluation complete! View the results in the LangSmith UI.")

if __name__ == "__main__":
    print("=== Hybrid RAG Agent Evaluation ===")
    run_evaluation()
