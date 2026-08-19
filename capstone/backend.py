from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from graph import graph
import json
from langchain_core.messages import HumanMessage
import os
import shutil
from ingest import ingest_documents

app = FastAPI(title="LangGraph Capstone API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InvokeRequest(BaseModel):
    message: str = None
    thread_id: str = "1"
    checkpoint_id: str = None

def serialize_message(m):
    if hasattr(m, "content"):
        return f"{m.__class__.__name__}: {m.content}"
    return str(m)

@app.post("/invoke")
def invoke_graph(req: InvokeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    if req.checkpoint_id:
        config["configurable"]["checkpoint_id"] = req.checkpoint_id
    
    # Check if the graph is currently paused
    state = graph.get_state(config)
    if state.next:
        result = graph.invoke(None, config) # Resume
    else:
        result = graph.invoke({"messages": [HumanMessage(content=req.message)]}, config)
        
    return {
        "draft": result.get("draft"), 
        "messages": [serialize_message(m) for m in result.get("messages", [])]
    }

@app.post("/stream")
def stream_graph(req: InvokeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    if req.checkpoint_id:
        config["configurable"]["checkpoint_id"] = req.checkpoint_id
    
    def generate():
        state = graph.get_state(config)
        inputs = None if state.next else {"messages": [HumanMessage(content=req.message)]}
        
        for stream_type, data in graph.stream(inputs, config=config, stream_mode=["updates", "messages"]):
            if stream_type == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") == "writer":
                    if hasattr(chunk, "content") and chunk.content:
                        yield f"data: {json.dumps({'token': chunk.content})}\n\n"
                continue
                
            if stream_type == "updates":
                serializable_update = {}
                for node, state_data in data.items():
                    node_data = {}
                    for k, v in state_data.items():
                        if k == "messages":
                            node_data[k] = [serialize_message(m) for m in v]
                        else:
                            node_data[k] = v
                    serializable_update[node] = node_data
                    
                yield f"data: {json.dumps(serializable_update)}\n\n"
            
        final_state = graph.get_state(config)
        if final_state.next:
            yield f"data: {json.dumps({'status': 'PAUSED', 'next': list(final_state.next)})}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/state/{thread_id}")
def get_current_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    return {
        "next": list(state.next),
        "values": {
            "research_topics": state.values.get("research_topics", []),
            "summaries": state.values.get("summaries", []),
            "draft": state.values.get("draft")
        }
    }

@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))
    serialized = []
    for h in history:
        serialized.append({
            "checkpoint_id": h.config["configurable"]["checkpoint_id"],
            "messages": [serialize_message(m) for m in h.values.get("messages", [])],
            "draft": h.values.get("draft")
        })
    return {"history": serialized}

@app.get("/graph_mermaid", response_class=PlainTextResponse)
def get_graph_mermaid():
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        return f"Error generating graph: {e}"

@app.post("/upload")
def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    raw_docs_dir = "data/raw_docs"
    os.makedirs(raw_docs_dir, exist_ok=True)
    file_path = os.path.join(raw_docs_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # Run ingestion in the background
    background_tasks.add_task(ingest_documents, raw_docs_dir, "data/chroma_db")
    
    return {"message": f"Successfully uploaded {file.filename} and queued for ingestion."}

if __name__ == "__main__":
    import uvicorn
    os.makedirs("data", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
