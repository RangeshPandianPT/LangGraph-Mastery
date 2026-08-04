import streamlit as st
import requests
import json
import sseclient
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="LangGraph Capstone", layout="wide", page_icon="🤖")
st.title("🤖 Multi-Agent Researcher Pro")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo_thread"

with st.sidebar:
    st.header("Settings & Tools")
    
    tab1, tab2 = st.tabs(["Controls", "Architecture"])
    
    with tab1:
        new_thread_id = st.text_input("Thread ID", value=st.session_state.thread_id)
        if new_thread_id != st.session_state.thread_id:
            st.session_state.thread_id = new_thread_id
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        
        if st.button("🔄 Check Graph State"):
            res = requests.get(f"{BACKEND_URL}/state/{st.session_state.thread_id}")
            if res.status_code == 200:
                st.json(res.json())
        
        if st.button("📜 Load Past History"):
            res = requests.get(f"{BACKEND_URL}/history/{st.session_state.thread_id}")
            if res.status_code == 200:
                history = res.json()["history"]
                for h in history:
                    with st.expander(f"Checkpoint: {h['checkpoint_id'][:8]}"):
                        for m in h["messages"]:
                            st.write(m)
                        if h.get("draft"):
                            st.success(h["draft"][:100] + "...")
                        if st.button(f"Fork & Resume from here", key=f"fork_{h['checkpoint_id']}"):
                            st.session_state.checkpoint_id = h['checkpoint_id']
                            st.session_state.chat_history.append({"role": "assistant", "content": f"🔄 **Time Travel**: Resuming from checkpoint {h['checkpoint_id'][:8]}..."})
                            st.rerun()
            else:
                st.error("Failed to load history")
                
        # If there's a draft in the current state, show download button
        state_res = requests.get(f"{BACKEND_URL}/state/{st.session_state.thread_id}")
        if state_res.status_code == 200:
            current_state = state_res.json()
            if current_state["values"].get("draft"):
                st.divider()
                st.download_button(
                    label="⬇️ Download Final Report (Markdown)",
                    data=current_state["values"]["draft"],
                    file_name="research_report.md",
                    mime="text/markdown"
                )
                
    with tab2:
        st.subheader("Graph Visualization")
        mermaid_res = requests.get(f"{BACKEND_URL}/graph_mermaid")
        if mermaid_res.status_code == 200:
            st.markdown(f"```mermaid\n{mermaid_res.text}\n```")
        else:
            st.error("Could not load graph visualization.")

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Check if paused for human approval
state_res = requests.get(f"{BACKEND_URL}/state/{st.session_state.thread_id}")
if state_res.status_code == 200:
    current_state = state_res.json()
    if "writer" in current_state.get("next", []):
        with st.chat_message("assistant"):
            st.warning("⏸️ **Graph Paused for Human Approval!** I am ready to write the report based on the gathered research.")
            with st.expander("View Gathered Research"):
                for summary in current_state["values"].get("summaries", []):
                    st.info(summary)
            
            if st.button("✅ Approve & Draft Report"):
                try:
                    payload = {"message": "", "thread_id": st.session_state.thread_id}
                    if hasattr(st.session_state, "checkpoint_id") and st.session_state.checkpoint_id:
                        payload["checkpoint_id"] = st.session_state.checkpoint_id
                        st.session_state.checkpoint_id = None # Clear after use
                        
                    req = requests.post(f"{BACKEND_URL}/stream", json=payload, stream=True)
                    client = sseclient.SSEClient(req)
                    
                    accumulated_text = ""
                    placeholder = st.empty()
                    for event in client.events():
                        data = json.loads(event.data)
                        if "status" in data and data["status"] == "PAUSED":
                            continue
                        
                        node_name = list(data.keys())[0]
                        state_update = data[node_name]
                        
                        if node_name == "writer" and "draft" in state_update:
                            accumulated_text += f"\n\n### Final Report Draft\n{state_update['draft']}\n\n"
                        elif node_name == "evaluator" and "evaluation" in state_update:
                            if state_update['evaluation'] == 'ACCEPT':
                                accumulated_text += f"\n**[Evaluator]**: ✅ Draft Accepted!\n"
                            else:
                                accumulated_text += f"\n**[Evaluator]**: 🔄 Revision Needed - {state_update['evaluation']}\n"
                        else:
                            messages = state_update.get("messages", [""])
                            if messages:
                                accumulated_text += f"\n*[{node_name}]*: {messages[-1]}\n"
                            
                        placeholder.markdown(accumulated_text)
                        
                    st.session_state.chat_history.append({"role": "assistant", "content": accumulated_text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error resuming: {e}")
        st.stop()

# Chat Input
if prompt := st.chat_input("What should the agents research?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        try:
            payload = {"message": prompt, "thread_id": st.session_state.thread_id}
            if hasattr(st.session_state, "checkpoint_id") and st.session_state.checkpoint_id:
                payload["checkpoint_id"] = st.session_state.checkpoint_id
                st.session_state.checkpoint_id = None # Clear after use
                
            req = requests.post(f"{BACKEND_URL}/stream", json=payload, stream=True)
            client = sseclient.SSEClient(req)
            
            accumulated_text = ""
            for event in client.events():
                data = json.loads(event.data)
                
                if "status" in data and data["status"] == "PAUSED":
                    accumulated_text += "\n\n⏸️ **Paused for Human Approval. Please review the research before drafting.**"
                    placeholder.markdown(accumulated_text)
                    break
                
                node_name = list(data.keys())[0]
                state_update = data[node_name]
                
                if node_name == "evaluator" and "evaluation" in state_update:
                    if state_update['evaluation'] == 'ACCEPT':
                        accumulated_text += f"\n**[Evaluator]**: ✅ Draft Accepted!\n"
                    else:
                        accumulated_text += f"\n**[Evaluator]**: 🔄 Revision Needed - {state_update['evaluation']}\n"
                
                msgs = state_update.get('messages', [])
                if msgs:
                    accumulated_text += f"\n*[{node_name}]*: {msgs[-1]}\n"
                    
                placeholder.markdown(accumulated_text)
                
            st.session_state.chat_history.append({"role": "assistant", "content": accumulated_text})
            st.rerun()
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
