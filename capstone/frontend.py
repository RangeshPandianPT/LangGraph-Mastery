import streamlit as st
import requests
import json
import sseclient

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="LangGraph Capstone", layout="wide")
st.title("🤖 Multi-Agent Researcher")

# Sidebar for Thread and History
with st.sidebar:
    st.header("Settings & Time Travel")
    thread_id = st.text_input("Thread ID", value="demo_thread")
    
    if st.button("Load History"):
        res = requests.get(f"{BACKEND_URL}/history/{thread_id}")
        if res.status_code == 200:
            history = res.json()["history"]
            for h in history:
                with st.expander(f"Checkpoint: {h['checkpoint_id'][:8]}"):
                    st.write(h["messages"])
                    if h.get("draft"):
                        st.success(h["draft"])
        else:
            st.error("Failed to load history")

st.markdown("### Chat with the Agent")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.markdown(f"**{msg['role']}**: {msg['content']}")

user_input = st.text_input("What should the agents research?")

if st.button("Send (Streamed)"):
    if user_input:
        st.session_state.chat_history.append({"role": "User", "content": user_input})
        st.markdown(f"**User**: {user_input}")
        
        response_placeholder = st.empty()
        
        # Call streaming endpoint
        try:
            req = requests.post(f"{BACKEND_URL}/stream", json={"message": user_input, "thread_id": thread_id}, stream=True)
            client = sseclient.SSEClient(req)
            
            accumulated_text = ""
            for event in client.events():
                data = json.loads(event.data)
                node_name = list(data.keys())[0]
                state_update = data[node_name]
                
                accumulated_text += f"\n*[{node_name}]*: {state_update.get('messages', [''])[0]}"
                response_placeholder.markdown(accumulated_text)
                
            st.session_state.chat_history.append({"role": "Agent", "content": accumulated_text})
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
