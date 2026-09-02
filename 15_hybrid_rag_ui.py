import streamlit as st
import sys
import os

# Add scripts directory to path to import langgraph_agent
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
try:
    from langgraph_agent import run_agent
except ImportError as e:
    st.error(f"Failed to import LangGraph Agent: {e}")
    st.stop()

st.set_page_config(
    page_title="Resume Knowledge Base AI",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #1e2329;
        border: 1px solid #30363d;
    }
    .avatar {
        width: 20%;
    }
    .avatar img {
        max-width: 50px;
        max-height: 50px;
        border-radius: 50%;
        object-fit: cover;
    }
    .message {
        width: 80%;
        padding: 0 1.5rem;
    }
    .title-area {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-area">
    <h1>💼 Candidate Search AI</h1>
</div>
<p style="color: #8b949e; font-size: 1.1em;">Powered by LangGraph, Neo4j, ChromaDB, and Qwen</p>
<hr>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Recruitment AI. Ask me to find candidates with specific skills, experience, or from specific companies."}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("E.g. Who has experience with Python and React?"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching Vector Database and Graph Database..."):
            try:
                result = run_agent(prompt)
                answer = result.get("answer", "I couldn't generate an answer.")
                
                # Show some metadata if we want
                resume_ids = result.get("resume_ids", [])
                
                st.markdown(answer)
                
                if resume_ids:
                    st.caption(f"Sources identified: {', '.join(resume_ids)}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.sidebar:
    st.header("Graph Info")
    st.info("This system uses a Hybrid RAG approach.")
    st.write("1. **Vector Search:** Finds semantically similar resumes in ChromaDB.")
    st.write("2. **Graph Search:** Retrieves structured nodes (Skills, Companies, Projects) from Neo4j.")
    st.write("3. **LangGraph Agent:** Orchestrates the workflow.")
    st.write("4. **Qwen LLM:** Synthesizes the final answer.")
    
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Recruitment AI. Ask me to find candidates with specific skills, experience, or from specific companies."}
        ]
        st.rerun()
