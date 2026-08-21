import pytest
from langchain_core.messages import HumanMessage, AIMessage
from graph import OverallState, supervisor, supervisor_router, map_research
from langgraph.graph import END
from langgraph.constants import Send
from unittest.mock import patch, MagicMock

def test_supervisor_routing_to_researcher():
    """Test that initially, supervisor routes to researcher."""
    state = OverallState(
        messages=[HumanMessage(content="What is LangGraph?")],
        research_topics=[],
        summaries=[],
        draft="",
        fact_check_result="",
        evaluation="",
        revision_count=0,
        next_agent=""
    )
    
    result = supervisor(state)
    assert result["next_agent"] == "researcher"

def test_supervisor_routing_to_writer():
    """Test that after research, it routes to writer."""
    state = OverallState(
        messages=[HumanMessage(content="What is LangGraph?")],
        research_topics=["LangGraph framework", "LangChain"],
        summaries=["Research summary here"],
        draft="",
        fact_check_result="",
        evaluation="",
        revision_count=0,
        next_agent=""
    )
    
    result = supervisor(state)
    assert result["next_agent"] == "writer"

def test_supervisor_routing_to_fact_checker():
    """Test routing to fact checker after draft generation."""
    state = OverallState(
        messages=[HumanMessage(content="What is LangGraph?")],
        research_topics=[],
        summaries=[],
        draft="Here is a draft.",
        fact_check_result="",
        evaluation="",
        revision_count=0,
        next_agent=""
    )
    
    result = supervisor(state)
    assert result["next_agent"] == "fact_checker"

def test_supervisor_routing_to_evaluator():
    """Test routing to evaluator after successful fact check."""
    state = OverallState(
        messages=[HumanMessage(content="What is LangGraph?")],
        research_topics=[],
        summaries=[],
        draft="Here is a draft.",
        fact_check_result="ACCEPT",
        evaluation="",
        revision_count=0,
        next_agent=""
    )
    
    result = supervisor(state)
    assert result["next_agent"] == "evaluator"
    
def test_supervisor_routing_to_finish():
    """Test routing to END after evaluator acceptance."""
    state = OverallState(
        messages=[HumanMessage(content="What is LangGraph?")],
        research_topics=[],
        summaries=[],
        draft="Here is a draft.",
        fact_check_result="ACCEPT",
        evaluation="ACCEPT",
        revision_count=0,
        next_agent=""
    )
    
    result = supervisor(state)
    assert result["next_agent"] == "FINISH"

def test_supervisor_router_finish():
    assert supervisor_router({"next_agent": "FINISH"}) == END

def test_supervisor_router_evaluator():
    assert supervisor_router({"next_agent": "evaluator"}) == "evaluator"

def test_supervisor_router_fact_checker():
    assert supervisor_router({"next_agent": "fact_checker"}) == "fact_checker"

def test_supervisor_router_researcher():
    assert supervisor_router({"next_agent": "researcher"}) == "researcher"

def test_supervisor_router_writer():
    assert supervisor_router({"next_agent": "writer"}) == "writer"

def test_map_research():
    """Test map_research fans out correctly."""
    state = OverallState(
        messages=[HumanMessage(content="What is LangGraph?")],
        research_topics=["Topic 1", "Topic 2"],
        summaries=[],
        draft="",
        fact_check_result="",
        evaluation="",
        revision_count=0,
        next_agent=""
    )
    
    sends = map_research(state)
    
    assert len(sends) == 4 # 2 topics + 1 local + 1 data analyst
    assert sends[0].node == "research_worker"
    assert sends[0].arg["topic"] == "Topic 1"
    
    assert sends[1].node == "research_worker"
    assert sends[1].arg["topic"] == "Topic 2"
    
    assert sends[2].node == "document_retriever"
    assert sends[2].arg["topic"] == "Local Documents"
    
    assert sends[3].node == "data_analyst"
    assert sends[3].arg["topic"] == "Data Analysis"

@patch("graph.llm.invoke")
def test_writer_node(mock_invoke):
    """Test the writer node uses LLM output as draft."""
    from graph import writer
    from langgraph.store.memory import InMemoryStore
    
    mock_invoke.return_value = MagicMock(content="Here is the final report.")
    
    state = OverallState(
        messages=[HumanMessage(content="Write a report")],
        research_topics=[],
        summaries=["Summary 1", "Summary 2"],
        draft="",
        fact_check_result="",
        evaluation="",
        revision_count=0,
        next_agent=""
    )
    
    store = InMemoryStore()
    config = {"configurable": {"user_id": "test_user"}}
    
    result = writer(state, config, store)
    
    assert result["draft"] == "Here is the final report."
    assert "Writer: Draft completed!" in result["messages"][0].content
    assert mock_invoke.called
