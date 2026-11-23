import pytest
from fastapi import HTTPException
from pydantic import ValidationError
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import (
    app,
    root,
    health_check,
    classify_question,
    ask_notes,
    summarize_notes,
    generate_quiz,
    run_agent,
    ClassificationRequest,
    ClassificationResponse,
    AskRequest,
    AskResponse,
    SummarizeRequest,
    SummarizeResponse,
    QuizRequest,
    QuizResponse,
    AgentRequest,
    AgentResponse,
)

def test_root_endpoint():
    res = root()
    assert res["message"] == "Welcome to AI Study Assistant API"
    assert res["health_check"] == "/health"
    assert res["docs_url"] == "/docs"
    assert res["dashboard_ui"] == "/dashboard"

def test_health_endpoint():
    res = health_check()
    assert res.status == "healthy"
    assert res.app_name == "AI Study Assistant"
    assert res.version in ["0.1.0", "0.2.0"]
    assert res.python_version is not None

def test_pydantic_schema_validations():
    # Valid classification request
    req = ClassificationRequest(question="What is a decorator in Python?")
    assert req.question == "What is a decorator in Python?"

    # Empty question (< 1 char) should raise ValidationError
    with pytest.raises(ValidationError):
        ClassificationRequest(question="")

    # Quiz num_questions out of range (> 10) should raise ValidationError
    with pytest.raises(ValidationError):
        QuizRequest(topic="Deep Learning", num_questions=25)

def test_classify_endpoint_with_trained_model():
    req = ClassificationRequest(question="How do list comprehensions work in Python?")
    resp = classify_question(req)
    assert isinstance(resp, ClassificationResponse)
    assert resp.category == "Python"
    assert resp.confidence > 0.70
    assert "Python" in resp.probabilities

def test_ask_endpoint_greeting():
    # Conversational greeting check: "hi" must return welcoming tutor message
    req = AskRequest(query="hi")
    resp = ask_notes(req)
    assert isinstance(resp, AskResponse)
    assert resp.found_in_notes is True
    assert resp.confidence == 1.0
    assert "AI Study Tutor" in resp.answer

def test_ask_rejects_whitespace_input():
    with pytest.raises(HTTPException) as exc:
        ask_notes(AskRequest(query=" "))

    assert exc.value.status_code == 400

def test_ask_endpoint_with_rag():
    # Module 3 RAG endpoint queries the indexed study notes
    req = AskRequest(query="What is the Global Interpreter Lock in Python?")
    resp = ask_notes(req)
    assert isinstance(resp, AskResponse)
    assert resp.found_in_notes is True
    assert "study notes" in resp.answer.lower()
    assert len(resp.citations) > 0

def test_ask_response_validation_error_is_server_error(monkeypatch):
    class FakeRagPipeline:
        def query(self, query):
            return {
                "answer": "A test response.",
                "found_in_notes": True,
                "confidence": 1.0,
                "citations": [{}],
                "question": query,
                "retrieved_count": 1,
            }

    class FakeDatabase:
        def add_conversation(self, **kwargs):
            return 1

    monkeypatch.setattr("api.main.get_rag_pipeline", lambda: FakeRagPipeline())
    monkeypatch.setattr("api.main.get_db", lambda: FakeDatabase())

    with pytest.raises(HTTPException) as exc:
        ask_notes(AskRequest(query="Tell me about a topic"))

    assert exc.value.status_code == 500
    assert "RAG query execution failed" in exc.value.detail

def test_agent_response_validation_error_is_server_error(monkeypatch):
    class FakeOrchestrator:
        def run(self, prompt):
            return {"success": True, "prompt": prompt}

    class FakeDatabase:
        def add_conversation(self, **kwargs):
            return 1

    monkeypatch.setattr("api.main.get_agent_orchestrator", lambda: FakeOrchestrator())
    monkeypatch.setattr("api.main.get_db", lambda: FakeDatabase())

    with pytest.raises(HTTPException) as exc:
        run_agent(AgentRequest(prompt="Tell me about a topic"))

    assert exc.value.status_code == 500
    assert "Agent workflow encountered an error" in exc.value.detail

def test_module4_agent_endpoints():
    # 1. Summarize Endpoint
    sum_resp = summarize_notes(SummarizeRequest(max_points=3))
    assert isinstance(sum_resp, SummarizeResponse)
    assert sum_resp.success is True
    assert sum_resp.tool_used == "Summary Tool"

    # 2. Quiz Endpoint
    quiz_resp = generate_quiz(QuizRequest(topic="Machine Learning", num_questions=2))
    assert isinstance(quiz_resp, QuizResponse)
    assert quiz_resp.success is True
    assert len(quiz_resp.questions) == 2

    # 3. Agent Endpoint
    agent_resp = run_agent(AgentRequest(prompt="Explain Python lists and generators"))
    assert isinstance(agent_resp, AgentResponse)
    assert agent_resp.success is True
    assert agent_resp.intent == "ANSWER"
