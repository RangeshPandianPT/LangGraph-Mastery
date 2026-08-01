import streamlit as st
import requests
import json
import sseclient
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="LangGraph Capstone", layout="wide")
st.title("🤖 Multi-Agent Researcher")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Settings & Time Travel")
    thread_id = st.text_input("Thread ID", value="demo_thread")
    
    if st.button("Check State"):
        res = requests.get(f"{BACKEND_URL}/state/{thread_id}")
        if res.status_code == 200:
            st.json(res.json())
    
    st.divider()
    if st.button("Load History"):
        res = requests.get(f"{BACKEND_URL}/history/{thread_id}")
        if res.status_code == 200:
            history = res.json()["history"]
            for h in history:
                with st.expander(f"Checkpoint: {h['checkpoint_id'][:8]}"):
                    for m in h["messages"]:
                        st.write(m)
                    if h.get("draft"):
                        st.success(h["draft"][:100] + "...")
        else:
            st.error("Failed to load history")

st.markdown("### Chat with the Agent")

for msg in st.session_state.chat_history:
    st.markdown(f"**{msg['role']}**: {msg['content']}")

# Handle Approval State
state_res = requests.get(f"{BACKEND_URL}/state/{thread_id}")
if state_res.status_code == 200:
    current_state = state_res.json()
    if "writer" in current_state.get("next", []):
        st.warning("⏸️ Graph is paused for Human Approval before drafting!")
        st.markdown("#### Research Gathered:")
        for summary in current_state["values"].get("summaries", []):
            st.info(summary)
            
        if st.button("Approve & Continue to Draft"):
            try:
                req = requests.post(f"{BACKEND_URL}/stream", json={"message": "", "thread_id": thread_id}, stream=True)
                client = sseclient.SSEClient(req)
                
                accumulated_text = ""
                response_placeholder = st.empty()
                for event in client.events():
                    data = json.loads(event.data)
                    if "status" in data and data["status"] == "PAUSED":
                        continue
                    
                    node_name = list(data.keys())[0]
                    state_update = data[node_name]
                    
                    if node_name == "writer" and "draft" in state_update:
                        accumulated_text += f"\n\n**Draft:**\n{state_update['draft']}"
                    else:
                        messages = state_update.get("messages", [""])
                        accumulated_text += f"\n*[{node_name}]*: {messages[-1] if messages else ''}"
                        
                    response_placeholder.markdown(accumulated_text)
                    
                st.session_state.chat_history.append({"role": "Agent", "content": accumulated_text})
                st.rerun()
            except Exception as e:
                st.error(f"Error resuming: {e}")
        st.stop()  # Stop execution so the user can't send new messages while paused

user_input = st.text_input("What should the agents research?")

if st.button("Send Request"):
    if user_input:
        st.session_state.chat_history.append({"role": "User", "content": user_input})
        
        response_placeholder = st.empty()
        
        try:
            req = requests.post(f"{BACKEND_URL}/stream", json={"message": user_input, "thread_id": thread_id}, stream=True)
            client = sseclient.SSEClient(req)
            
            accumulated_text = ""
            for event in client.events():
                data = json.loads(event.data)
                
                if "status" in data and data["status"] == "PAUSED":
                    accumulated_text += "\n\n⚠️ **Paused for Human Approval. Please review above.**"
                    response_placeholder.markdown(accumulated_text)
                    break
                
                node_name = list(data.keys())[0]
                state_update = data[node_name]
                
                msgs = state_update.get('messages', [])
                if msgs:
                    accumulated_text += f"\n*[{node_name}]*: {msgs[-1]}"
                    
                response_placeholder.markdown(accumulated_text)
                
            st.session_state.chat_history.append({"role": "Agent", "content": accumulated_text})
            st.rerun()
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
