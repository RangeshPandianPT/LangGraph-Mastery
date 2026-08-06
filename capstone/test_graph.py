import pytest
from langchain_core.messages import HumanMessage, AIMessage
from graph import OverallState, supervisor

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
