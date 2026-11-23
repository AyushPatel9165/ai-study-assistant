import os
import sys
from pathlib import Path
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.intent_detector import IntentDetector
from src.agents.tools.answer_tool import AnswerTool
from src.agents.tools.summary_tool import SummaryTool
from src.agents.tools.quiz_tool import QuizTool
from src.agents.orchestrator import AgentOrchestrator
from src.rag.pipeline import RAGPipeline
from api.main import (
    summarize_notes,
    generate_quiz,
    run_agent,
    SummarizeRequest,
    SummarizeResponse,
    QuizRequest,
    QuizResponse,
    AgentRequest,
    AgentResponse
)

@pytest.fixture(scope="module")
def rag_pipeline():
    base_dir = Path(__file__).resolve().parent.parent
    idx_path = str(base_dir / "models" / "faiss_index.bin")
    meta_path = str(base_dir / "models" / "faiss_chunks.json")
    return RAGPipeline(index_path=idx_path, metadata_path=meta_path, confidence_threshold=0.35)

@pytest.fixture(scope="module")
def intent_detector():
    return IntentDetector()

def test_intent_detector_quiz(intent_detector):
    prompts = [
        "Create 5 MCQs from Python notes",
        "Generate a quiz on machine learning",
        "Test me on regularization techniques",
        "Can you make 3 practice questions?"
    ]
    for p in prompts:
        res = intent_detector.detect_intent(p)
        assert res["intent"] == "QUIZ"
        assert res["confidence"] >= 0.70

def test_intent_detector_summary(intent_detector):
    prompts = [
        "Summarize my ML notes",
        "Give me an overview of sample_study_notes.txt",
        "What are the key points and takeaways from the uploaded notes?",
        "Provide a brief synopsis of the document"
    ]
    for p in prompts:
        res = intent_detector.detect_intent(p)
        assert res["intent"] == "SUMMARY"
        assert res["confidence"] >= 0.70

def test_intent_detector_answer(intent_detector):
    prompts = [
        "Explain Python lists",
        "What is the role of Global Interpreter Lock?",
        "How does gradient descent work?",
        "What is supervised learning?"
    ]
    for p in prompts:
        res = intent_detector.detect_intent(p)
        assert res["intent"] == "ANSWER"

def test_summary_tool_execution(rag_pipeline):
    tool = SummaryTool(vector_store=rag_pipeline.vector_store)
    result = tool.execute(max_points=3)
    assert result["success"] is True
    assert result["tool_used"] == "Summary Tool"
    assert "executive_summary" in result
    assert len(result["key_takeaways"]) > 0

def test_quiz_tool_execution(rag_pipeline):
    tool = QuizTool(vector_store=rag_pipeline.vector_store)
    result = tool.execute(topic="Python", num_questions=3)
    assert result["success"] is True
    assert result["tool_used"] == "Quiz Tool"
    assert len(result["questions"]) == 3
    q1 = result["questions"][0]
    assert "question" in q1
    assert "options" in q1
    assert "correct_option" in q1
    assert "explanation" in q1

def test_agent_orchestrator_routing(rag_pipeline):
    orchestrator = AgentOrchestrator(
        rag_pipeline=rag_pipeline,
        vector_store=rag_pipeline.vector_store
    )

    # 1. Test Quiz Route
    res_quiz = orchestrator.run("Generate 2 MCQs on Python")
    assert res_quiz["success"] is True
    assert res_quiz["intent"] == "QUIZ"
    assert res_quiz["tool_used"] == "Quiz Tool"
    assert len(res_quiz["execution_trace"]) >= 4

    # 2. Test Summary Route
    res_sum = orchestrator.run("Summarize the study notes")
    assert res_sum["success"] is True
    assert res_sum["intent"] == "SUMMARY"
    assert res_sum["tool_used"] == "Summary Tool"

    # 3. Test Answer Route
    res_ans = orchestrator.run("What is the Global Interpreter Lock in Python?")
    assert res_ans["success"] is True
    assert res_ans["intent"] == "ANSWER"
    assert res_ans["tool_used"] == "Answer Tool"

def test_api_agent_endpoint_handlers():
    # 1. Direct handler test for POST /summarize
    resp_sum = summarize_notes(SummarizeRequest(max_points=3))
    assert isinstance(resp_sum, SummarizeResponse)
    assert resp_sum.success is True
    assert resp_sum.tool_used == "Summary Tool"

    # 2. Direct handler test for POST /quiz
    resp_quiz = generate_quiz(QuizRequest(topic="Machine Learning", num_questions=2))
    assert isinstance(resp_quiz, QuizResponse)
    assert resp_quiz.success is True
    assert len(resp_quiz.questions) == 2

    # 3. Direct handler test for POST /agent
    resp_agent = run_agent(AgentRequest(prompt="Create 3 MCQs from Python notes"))
    assert isinstance(resp_agent, AgentResponse)
    assert resp_agent.success is True
    assert resp_agent.intent == "QUIZ"
    assert resp_agent.tool_used == "Quiz Tool"
