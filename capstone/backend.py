from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from graph import graph
import json

app = FastAPI(title="LangGraph Capstone API")

class InvokeRequest(BaseModel):
    message: str
    thread_id: str = "1"

@app.post("/invoke")
def invoke_graph(req: InvokeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke({"messages": [req.message]}, config)
    return {"draft": result.get("draft"), "messages": result.get("messages")}

@app.post("/stream")
def stream_graph(req: InvokeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    
    def generate():
        for update in graph.stream({"messages": [req.message]}, config=config, stream_mode="updates"):
            yield f"data: {json.dumps(update)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))
    serialized = []
    for h in history:
        serialized.append({
            "checkpoint_id": h.config["configurable"]["checkpoint_id"],
            "messages": h.values.get("messages", []),
            "draft": h.values.get("draft")
        })
    return {"history": serialized}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
