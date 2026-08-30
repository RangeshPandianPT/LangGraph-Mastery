import asyncio
import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# Note: Requires `pip install langgraph-checkpoint-postgres psycopg-pool`
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# --- 1. Define State ---
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- 2. Define Node ---
async def chatbot(state: State):
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# --- 3. Build Graph ---
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Database connection string (replace with your actual Postgres credentials)
# E.g., "postgres://user:password@localhost:5432/langgraph"
DB_URI = os.getenv("POSTGRES_URI", "postgres://postgres:postgres@localhost:5432/postgres")

async def main():
    print(f"Connecting to Postgres at {DB_URI}...")
    
    try:
        # Create an async connection pool
        # For a sync version, use PostgresSaver and psycopg.Connection
        from psycopg_pool import AsyncConnectionPool
        
        async with AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=20,
        ) as pool:
            # Initialize PostgresSaver
            checkpointer = AsyncPostgresSaver(pool)
            
            # Setup the database tables (only needs to be run once)
            await checkpointer.setup()
            
            # Compile graph with the Postgres checkpointer
            app = builder.compile(checkpointer=checkpointer)
            
            # Setup config with a thread ID (this represents the conversation session)
            config = {"configurable": {"thread_id": "postgres_test_thread_1"}}
            
            print("\n--- Conversation Turn 1 ---")
            input_message = HumanMessage(content="Hi! My name is LangGraph Developer.")
            async for event in app.astream({"messages": [input_message]}, config=config):
                for k, v in event.items():
                    print(f"Node '{k}' executed.")
            
            print("\n--- Conversation Turn 2 (Testing Memory) ---")
            input_message_2 = HumanMessage(content="What is my name?")
            async for event in app.astream({"messages": [input_message_2]}, config=config):
                for k, v in event.items():
                    print(f"Node '{k}': {v['messages'][-1].content}")
                    
    except Exception as e:
        print(f"\n[!] Failed to connect or run Postgres checkpointing.")
        print(f"[!] Ensure you have a running Postgres instance and set POSTGRES_URI.")
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
