# LangGraph Basics: From Scratch to Mastery

Welcome to your LangGraph learning journey! LangGraph is a powerful framework built on top of LangChain for creating stateful, multi-actor applications with Large Language Models (LLMs). It allows you to model your agent workflows as graphs.

## Why LangGraph?
While LangChain's Expression Language (LCEL) is great for linear pipelines, real-world AI agents often require:
1. **Cycles/Loops:** The ability for an agent to think, act, observe, and think again.
2. **State Management:** Remembering context across multiple steps.
3. **Controllability:** Explicitly defining how different parts of your system interact (routing, conditionals, human-in-the-loop).

LangGraph solves this by modeling workflows as **Stateful Graphs**.

## Core Concepts

1. **State (`StateGraph`)**: 
   - Every graph is initialized with a "State" (usually a Python `TypedDict` or Pydantic model). 
   - This state is passed around to every node. 
   - Nodes return *updates* to this state, not the whole state.
   
2. **Nodes (`add_node`)**: 
   - Python functions or LLM chains that do the actual work. 
   - They receive the current state, perform an action, and return a dictionary of state updates.

3. **Edges (`add_edge`)**: 
   - Define the flow. They connect nodes together (e.g., after Node A, go to Node B).
   - **Normal Edges:** Always go from Node A to Node B.
   - **Conditional Edges:** Run a function to decide where to go next based on the current state.

4. **START and END**: 
   - Special nodes representing the entry and exit points of your graph.

## Learning Path in this Directory
1. **`01_basic_agent.py`**: Learn the absolute basics. Defining a state, nodes, and simple edges. No LLM required.
2. **`02_conditional_routing.py`**: Learn how to use conditional edges to make decisions and route workflows dynamically.

## Getting Started
First, install the requirements:
```bash
pip install -r requirements.txt
```

Then run the scripts sequentially:
```bash
python 01_basic_agent.py
python 02_conditional_routing.py
```
