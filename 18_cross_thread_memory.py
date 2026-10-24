"""
18_cross_thread_memory.py: Long-Term Cross-Thread Memory using LangGraph Store

What you will learn:
1. The architectural difference between Checkpointers and Store:
   - Checkpointer: Thread-scoped short-term memory (tied to a single `thread_id`).
   - Store (BaseStore / InMemoryStore): Cross-thread long-term semantic memory
     accessible across completely different sessions and threads.
2. Structuring hierarchical namespaces (e.g., ("users", user_id, "memories")).
3. Accessing the store directly inside graph nodes via `*, store: BaseStore`.
4. How an agent in a brand-new conversation thread automatically remembers
   user facts learned from a previous thread.
"""

from typing import TypedDict, List, Optional
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

# --- 1. State Definition ---
class ChatState(TypedDict):
    user_id: str
    message: str
    recalled_context: List[str]
    response: str

# --- 2. Node Functions with Store Injection ---

def recall_user_memories(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Queries the cross-thread store for facts associated with this user."""
    user_id = state["user_id"]
    namespace = ("users", user_id, "facts")
    print(f"\n[Store: Recall] Searching long-term memories in namespace: {namespace}...")
    
    # Search all memories saved under this user's namespace
    stored_items = store.search(namespace)
    recalled = []
    for item in stored_items:
        fact = item.value.get("text")
        if fact:
            recalled.append(fact)
            
    print(f"[Store: Recall] Retrieved {len(recalled)} long-term memory records.")
    for r in recalled:
        print(f"   * {r}")
        
    return {"recalled_context": recalled}

def assistant_respond(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Generates an answer using both the active message and recalled long-term context."""
    user_id = state["user_id"]
    msg = state["message"]
    memories = state.get("recalled_context", [])
    
    print(f"\n[Assistant] Processing request for user '{user_id}'...")
    
    # Personalize response using recalled memories if available
    context_str = "; ".join(memories) if memories else "None"
    
    if "scaffold" in msg.lower() or "microservice" in msg.lower() or "backend" in msg.lower():
        if any("FastAPI" in m for m in memories):
            reply = (
                f"Hello {user_id.capitalize()}! Since you prefer Python & Async FastAPI (from long-term memory),\n"
                f"here is your asynchronous FastAPI microservice template:\n\n"
                f"```python\n"
                f"from fastapi import FastAPI\n\n"
                f"app = FastAPI(title='Enterprise Async Microservice')\n\n"
                f"@app.get('/health')\n"
                f"async def health_check():\n"
                f"    return {{'status': 'healthy', 'user': '{user_id}'}}\n"
                f"```"
            )
        else:
            reply = "Here is a standard generic microservice template."
    elif "who am i" in msg.lower() or "remember" in msg.lower():
        reply = f"Based on my long-term memory about you: {context_str}"
    else:
        reply = f"Acknowledged, {user_id.capitalize()}. I've processed your message: '{msg}'"
        
    return {"response": reply}

def extract_and_save_memories(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Analyzes message for user traits or preferences and persists them to the Store."""
    user_id = state["user_id"]
    msg = state["message"]
    namespace = ("users", user_id, "facts")
    
    new_facts = []
    # Simple semantic rule-based extraction for tutorial clarity
    if "python" in msg.lower() and "fastapi" in msg.lower():
        new_facts.append("Prefers Python with Async FastAPI for backend architecture.")
    if "ai engineer" in msg.lower():
        new_facts.append("Role: AI Engineer working with stateful agents.")
    if "alice" in msg.lower():
        new_facts.append("User name is Alice.")
        
    for fact in new_facts:
        key = str(uuid.uuid4())[:8]
        print(f"[Store: Persist] Saving new long-term memory -> key='{key}': '{fact}'")
        store.put(namespace, key, {"text": fact})
        
    return {}

# --- 3. Build Graph with Checkpointer + Store ---
def build_agent_with_memory():
    builder = StateGraph(ChatState)
    
    builder.add_node("recall", recall_user_memories)
    builder.add_node("respond", assistant_respond)
    builder.add_node("save_memory", extract_and_save_memories)
    
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "respond")
    builder.add_edge("respond", "save_memory")
    builder.add_edge("save_memory", END)
    
    # Initialize checkpointer (short-term thread state) and store (long-term memory)
    checkpointer = MemorySaver()
    store = InMemoryStore()
    
    # Notice we pass BOTH checkpointer and store to compile()
    graph = builder.compile(checkpointer=checkpointer, store=store)
    return graph, store

# --- 4. Main Demonstration ---
def main():
    print("=" * 70)
    print("LangGraph: Cross-Thread Long-Term Memory (Store)")
    print("=" * 70)
    
    agent, store = build_agent_with_memory()
    
    # -------------------------------------------------------------
    # Session 1: Thread "session_thread_1"
    # User shares information about themselves
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> SESSION 1 (Thread ID: 'session_thread_1') <<<")
    print("=" * 50)
    
    config_1 = {"configurable": {"thread_id": "session_thread_1"}}
    input_1 = {
        "user_id": "alice",
        "message": "Hi, I am Alice. I am an AI Engineer and I exclusively build using Python and FastAPI.",
        "recalled_context": [],
        "response": ""
    }
    
    result_1 = agent.invoke(input_1, config=config_1)
    print(f"\nAgent: {result_1['response']}")
    
    # -------------------------------------------------------------
    # Session 2: Thread "session_thread_2" (COMPLETELY NEW THREAD!)
    # In standard checkpointers, the agent would have 0 memory here.
    # But with LangGraph Store, it recalls facts about Alice!
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> SESSION 2 (Thread ID: 'session_thread_2' - BRAND NEW CONVERSATION) <<<")
    print("=" * 50)
    
    config_2 = {"configurable": {"thread_id": "session_thread_2"}}
    input_2 = {
        "user_id": "alice",
        "message": "Can you scaffold a microservice backend template for me?",
        "recalled_context": [],
        "response": ""
    }
    
    result_2 = agent.invoke(input_2, config=config_2)
    print(f"\nAgent:\n{result_2['response']}")
    
    # -------------------------------------------------------------
    # Inspecting Store Directly
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> DIRECT INSPECTION OF STORE CONTENTS <<<")
    print("=" * 50)
    alice_memories = store.search(("users", "alice", "facts"))
    print(f"Total memories stored for user 'alice': {len(alice_memories)}")
    for m in alice_memories:
        print(f"Key: {m.key} | Value: {m.value}")

if __name__ == "__main__":
    main()
