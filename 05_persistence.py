import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. State definition
class State(TypedDict):
    messages: Annotated[list[str], operator.add]

# 2. Node
def chat_node(state: State):
    # A simple mock node that just echoes and acknowledges the message
    latest_message = state["messages"][-1]
    response = f"Echo: I received '{latest_message}'"
    return {"messages": [response]}

# 3. Build Graph
builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# IMPORTANT: Initialize the checkpointer
# MemorySaver stores state in memory. For production, you'd use PostgresSaver or SqliteSaver.
memory = MemorySaver()

# Pass the checkpointer to compile()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    # We must provide a 'thread_id' in the config to identify the session/conversation.
    config = {"configurable": {"thread_id": "thread_1"}}
    
    print("--- Turn 1 ---")
    # We run the graph with an initial message
    state1 = graph.invoke({"messages": ["Hello! My name is Alice."]}, config=config)
    print("Messages after Turn 1:")
    for m in state1["messages"]:
        print(f" - {m}")
        
    print("\n--- Turn 2 ---")
    # In turn 2, we ONLY pass the new message. 
    # The checkpointer remembers the previous messages from Turn 1!
    state2 = graph.invoke({"messages": ["What is my name?"]}, config=config)
    print("Messages after Turn 2 (Notice the memory!):")
    for m in state2["messages"]:
        print(f" - {m}")
        
    print("\n--- Turn 3 (Different Thread) ---")
    # If we use a different thread_id, it's a completely blank state
    config2 = {"configurable": {"thread_id": "thread_2"}}
    state3 = graph.invoke({"messages": ["What is my name?"]}, config=config2)
    print("Messages after Turn 3 (New Thread):")
    for m in state3["messages"]:
        print(f" - {m}")
