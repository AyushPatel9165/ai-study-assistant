import os
import sys
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification.predict import QuestionClassifier
from src.rag.pipeline import RAGPipeline
from src.agents.orchestrator import AgentOrchestrator
from src.agents.tools.summary_tool import SummaryTool
from src.agents.tools.quiz_tool import QuizTool
from src.guardrails.security import PromptInjectionGuard, SensitiveDataFilter, RateLimiter
from src.database.db import DatabaseManager

app = FastAPI(
    title="AI Study Assistant API",
    description="Production-grade API for Question Classification, RAG Notes Q&A, AI Agents, and Study Dashboard.",
    version="0.2.0"
)

# Enable CORS for Frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singletons
_classifier: Optional[QuestionClassifier] = None
_rag_pipeline: Optional[RAGPipeline] = None
_agent_orchestrator: Optional[AgentOrchestrator] = None
_db: Optional[DatabaseManager] = None

# Guardrails singletons
injection_guard = PromptInjectionGuard()
data_filter = SensitiveDataFilter()
rate_limiter = RateLimiter(max_requests_per_minute=120)

def get_db() -> DatabaseManager:
    global _db
    if _db is None:
        _db = DatabaseManager()
    return _db

def get_classifier() -> QuestionClassifier:
    global _classifier
    if _classifier is None:
        base_dir = Path(__file__).resolve().parent.parent
        model_path = str(base_dir / "models" / "question_classifier.pt")
        vec_path = str(base_dir / "models" / "tfidf_vectorizer.pkl")
        enc_path = str(base_dir / "models" / "label_encoder.pkl")
        _classifier = QuestionClassifier(
            model_path=model_path,
            vectorizer_path=vec_path,
            encoder_path=enc_path
        )
    return _classifier

def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        base_dir = Path(__file__).resolve().parent.parent
        idx_path = str(base_dir / "models" / "faiss_index.bin")
        meta_path = str(base_dir / "models" / "faiss_chunks.json")
        _rag_pipeline = RAGPipeline(
            index_path=idx_path,
            metadata_path=meta_path,
            confidence_threshold=0.40
        )
    return _rag_pipeline

def get_agent_orchestrator() -> AgentOrchestrator:
    global _agent_orchestrator
    if _agent_orchestrator is None:
        rag = get_rag_pipeline()
        _agent_orchestrator = AgentOrchestrator(
            rag_pipeline=rag,
            vector_store=rag.vector_store
        )
    return _agent_orchestrator

# Security Helper
def verify_and_sanitize_input(text: str, client_ip: str = "127.0.0.1") -> str:
    # 0. Empty / Whitespace Validation (Must return 400, not 500)
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input query or prompt cannot be empty or whitespace only."
        )

    # 1. Rate Limiting Check
    allowed, _ = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before making more requests."
        )

    # 2. Prompt Injection Guard
    is_safe, reason = injection_guard.check_safety(text)
    if not is_safe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=reason
        )

    # 3. Sensitive Data Redaction
    sanitized, _ = data_filter.sanitize(text)
    return sanitized

# Pydantic Schemas
class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    python_version: str

class ClassificationRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class ClassificationResponse(BaseModel):
    question: str
    category: str
    confidence: float
    probabilities: Dict[str, float]
    cleaned_question: str
    top_features: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None

class CitationItem(BaseModel):
    document_name: str
    page_number: int
    relevance_score: float
    snippet: str

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(default="default_session")
    previous_query: Optional[str] = None

class AskResponse(BaseModel):
    question: str
    answer: str
    found_in_notes: bool
    confidence: float
    citations: List[CitationItem]
    retrieved_count: int

class UploadResponse(BaseModel):
    message: str
    document_name: str
    pages_processed: int
    chunks_created: int
    total_indexed_chunks: int

class SummarizeRequest(BaseModel):
    document_name: Optional[str] = Field(default=None)
    max_points: int = Field(default=5, ge=1, le=20)

class SummarizeResponse(BaseModel):
    success: bool
    tool_used: str
    document_name: Optional[str] = None
    total_chunks_analyzed: Optional[int] = None
    executive_summary: Optional[str] = None
    key_takeaways: Optional[List[str]] = None
    summary_points_count: Optional[int] = None
    error: Optional[str] = None

class QuizQuestionItem(BaseModel):
    id: int
    question: str
    options: List[str]
    correct_option: str
    explanation: str

class QuizRequest(BaseModel):
    topic: Optional[str] = Field(default=None)
    num_questions: int = Field(default=3, ge=1, le=10)

class QuizResponse(BaseModel):
    success: bool
    tool_used: str
    topic: Optional[str] = None
    num_questions: Optional[int] = None
    questions: Optional[List[QuizQuestionItem]] = None
    error: Optional[str] = None

class QuizSubmitRequest(BaseModel):
    topic: str
    score: int
    total_questions: int

class QuizSubmitResponse(BaseModel):
    success: bool
    message: str
    score: int
    total_questions: int
    percentage: float

class AgentRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(default="default_session")

class AgentResponse(BaseModel):
    success: bool
    prompt: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    tool_used: Optional[str] = None
    extracted_parameters: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    agent_response: Optional[str] = None
    execution_trace: List[str]
    error: Optional[str] = None

class MetricsResponse(BaseModel):
    total_documents: int
    total_questions_asked: int
    total_quizzes_generated: int
    average_quiz_score: float

@app.get("/", tags=["System"])
def root() -> Dict[str, str]:
    return {
        "message": "Welcome to AI Study Assistant API",
        "dashboard_ui": "/dashboard",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/dashboard", include_in_schema=False)
def serve_dashboard():
    dashboard_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard frontend not found.")
    return FileResponse(str(dashboard_path))

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        app_name="AI Study Assistant",
        version="0.2.0",
        python_version=sys.version.split()[0]
    )

@app.get("/database/status", tags=["System"])
def get_database_status() -> Dict[str, Any]:
    """Returns database status and current engine (MySQL or SQLite)."""
    db = get_db()
    info = db.get_engine_info()
    metrics = db.get_metrics()
    info.update({
        "total_documents": metrics["total_documents"],
        "total_conversations": metrics["total_questions_asked"],
        "total_quizzes": metrics["total_quizzes_generated"]
    })
    return info

@app.get("/metrics", response_model=MetricsResponse, tags=["Dashboard"])
def get_dashboard_metrics() -> MetricsResponse:
    """Returns aggregated usage statistics for the frontend dashboard cards."""
    db = get_db()
    metrics = db.get_metrics()
    return MetricsResponse(
        total_documents=metrics["total_documents"],
        total_questions_asked=metrics["total_questions_asked"],
        total_quizzes_generated=metrics["total_quizzes_generated"],
        average_quiz_score=metrics["average_quiz_score"]
    )

@app.get("/dashboard/summary", tags=["Dashboard"])
def get_dashboard_summary() -> Dict[str, Any]:
    """Returns comprehensive analytics, metrics, recent activities, and system info for the dashboard."""
    db = get_db()
    summary = db.get_dashboard_summary()
    engine_name = "MySQL" if db.engine == "mysql" else "SQLite 3"
    summary["system_info"] = {
        "status": "Online",
        "model": "QuestionClassifierMLP (PyTorch)",
        "model_accuracy": "96.15%",
        "embeddings": "sentence-transformers/all-MiniLM-L6-v2 (384-d)",
        "vector_store": "FAISS IndexFlatIP",
        "anti_hallucination_threshold": 0.40,
        "database": engine_name,
        "database_engine": db.engine,
        "version": "0.2.0"
    }
    return summary

@app.post("/classify", response_model=ClassificationResponse, tags=["Module 1: Classification"])
def classify_question(request: ClassificationRequest) -> ClassificationResponse:
    """Classify student questions into Python, AI, or ML using the PyTorch MLP model."""
    cleaned_input = verify_and_sanitize_input(request.question)
    try:
        classifier = get_classifier()
        result = classifier.predict(cleaned_input)
        return ClassificationResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification inference failed: {str(e)}"
        )

@app.post("/upload", response_model=UploadResponse, tags=["Module 3: Ingestion"])
def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and index study notes in PDF or TXT format into the FAISS vector database."""
    filename = file.filename or "uploaded_note.txt"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only .pdf and .txt files are allowed."
        )

    base_dir = Path(__file__).resolve().parent.parent
    upload_dir = base_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / filename

    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(destination)
        rag = get_rag_pipeline()
        index_stats = rag.index_document(str(destination))

        # Persist to SQLite
        db = get_db()
        db.add_document(
            filename=filename,
            file_type=ext.replace(".", "").upper(),
            file_size_bytes=file_size,
            chunks_count=index_stats["chunks_created"]
        )

        return UploadResponse(
            message=f"Document '{filename}' successfully uploaded and indexed.",
            **index_stats
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index document: {str(e)}"
        )

@app.get("/documents", tags=["Module 3: Ingestion"])
def list_uploaded_documents() -> List[Dict[str, Any]]:
    """List all study notes registered in the SQLite database."""
    db = get_db()
    return db.get_documents()

@app.post("/documents/sample", response_model=UploadResponse, tags=["Module 3: Ingestion"])
def load_sample_document() -> UploadResponse:
    """Loads and re-indexes the bundled sample_study_notes.txt into FAISS and SQLite."""
    base_dir = Path(__file__).resolve().parent.parent
    sample_file = base_dir / "data" / "raw" / "sample_study_notes.txt"
    if not sample_file.exists():
        raise HTTPException(status_code=404, detail="Sample study notes file not found.")

    file_size = os.path.getsize(sample_file)
    rag = get_rag_pipeline()
    index_stats = rag.index_document(str(sample_file))

    db = get_db()
    db.add_document(
        filename="sample_study_notes.txt",
        file_type="TXT",
        file_size_bytes=file_size,
        chunks_count=index_stats["chunks_created"]
    )
    return UploadResponse(
        message="Sample study notes successfully indexed into FAISS vector database.",
        **index_stats
    )

@app.get("/documents/{filename}/chunks", tags=["Module 3: Ingestion"])
def get_document_chunks(filename: str) -> List[Dict[str, Any]]:
    """Inspect all chunks extracted and embedded for a specific document."""
    rag = get_rag_pipeline()
    target_lower = filename.lower()
    matched_chunks = [
        {
            "chunk_id": c.chunk_id,
            "page_number": c.page_number,
            "char_length": len(c.text),
            "text": c.text
        }
        for c in rag.vector_store.chunks
        if c.document_name.lower() == target_lower
    ]
    return matched_chunks

@app.delete("/documents/{filename}", tags=["Module 3: Ingestion"])
def delete_document(filename: str) -> Dict[str, Any]:
    """Delete document from SQLite, FAISS vector index, and uploads directory."""
    db = get_db()
    deleted = db.delete_document(filename)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found."
        )

    # Remove from FAISS Vector Store
    try:
        rag = get_rag_pipeline()
        rag.remove_document(filename)
    except Exception as e:
        print(f"[WARN] Failed to remove chunks from FAISS: {e}")

    # Remove physical upload file if present
    base_dir = Path(__file__).resolve().parent.parent
    upload_file = base_dir / "uploads" / filename
    if upload_file.exists():
        try:
            upload_file.unlink()
        except Exception:
            pass

    return {
        "success": True,
        "message": f"Document '{filename}' successfully deleted from database and vector index."
    }

import re

GREETING_REGEX = re.compile(
    r"^(h+[iey]+[a-z]*|h+e+l+o+[a-z]*|h+l+o+[a-z]*|y+o+|n+a+m+a+s+t+e+|s+u+p+|s+u+n+o+|h+a+i+)\b"
    r"|^(who\s+are\s+you|what\s+can\s+you\s+do|help|how\s+does\s+this\s+work|kaise\s+ho|aap\s+kaun\s+ho|kuch\s+bhi)",
    re.IGNORECASE
)

SYLLABUS_QUERIES = {
    "notes", "study notes", "what notes do you have", "what notes are available",
    "show notes", "list notes", "kya notes hai", "kya notes hain",
    "syllabus", "topics", "all notes", "help notes", "view notes"
}

def is_conversational_greeting(query: str) -> bool:
    q = query.lower().strip().rstrip(".!? \t\n")
    return bool(GREETING_REGEX.search(q))

def is_syllabus_query(query: str) -> bool:
    q = query.lower().strip().rstrip(".!? \t\n")
    return q in SYLLABUS_QUERIES

@app.post("/ask", response_model=AskResponse, tags=["Module 3: RAG"])
def ask_notes(request: AskRequest) -> AskResponse:
    """Answer student questions grounded strictly in uploaded study notes using FAISS semantic search."""
    cleaned_query = verify_and_sanitize_input(request.query)

    # 1. Conversational Greeting Guard: Give a warm, helpful tutor welcome instead of cold out-of-scope rejection
    if is_conversational_greeting(cleaned_query):
        tutor_reply = (
            "Hello! I am your AI Study Tutor.\n\n"
            "I am ready to help you learn and understand topics from your study notes. You can ask me technical questions such as:\n"
            "• 'What is the Global Interpreter Lock (GIL) in Python?'\n"
            "• 'Explain overfitting and how L1/L2 regularization prevents it.'\n"
            "• 'How does A* search evaluate nodes with heuristics?'\n\n"
            "Or ask for chapter notes directly (e.g. 'ml notes', 'python notes', 'ai notes')!\n\n"
            "Other features you can explore:\n"
            "• Summarizer: Click 'Summarizer' in the sidebar for structured chapter takeaways.\n"
            "• Practice Quiz: Click 'Practice Quiz' to test yourself with MCQs.\n"
            "• My Documents: Upload your own lecture PDFs or TXT notes anytime!"
        )
        db = get_db()
        db.add_conversation(
            session_id=request.session_id,
            role="student",
            query=cleaned_query,
            answer=tutor_reply,
            confidence=1.0,
            tool_used="Tutor Greeting"
        )
        return AskResponse(
            question=request.query,
            answer=tutor_reply,
            found_in_notes=True,
            confidence=1.0,
            citations=[],
            retrieved_count=0
        )

    # 2. General Syllabus / Notes Query Guard
    if is_syllabus_query(cleaned_query):
        syllabus_reply = (
            "Here are the study notes currently available in your AI Study Assistant:\n\n"
            "1. Chapter 1: Python Advanced Internals & Memory Management\n"
            "   • Reference counting & Cyclic Garbage Collector (Generations 0, 1, 2)\n"
            "   • Global Interpreter Lock (GIL) and CPU-bound thread concurrency\n"
            "   • Python Generators and the 'yield' keyword for lazy evaluation\n\n"
            "2. Chapter 2: Machine Learning Optimization & Regularization\n"
            "   • Supervised loss minimization & Gradient Descent optimization\n"
            "   • Overfitting: high variance vs low bias on unseen validation sets\n"
            "   • L1 Regularization (Lasso) for sparsity and feature selection\n"
            "   • L2 Regularization (Ridge) for shrinking weights of correlated features\n\n"
            "3. Chapter 3: Artificial Intelligence Search & Heuristics\n"
            "   • State spaces and goal search paths\n"
            "   • A* search evaluation function: f(n) = g(n) + h(n)\n"
            "   • Admissible and consistent (monotonic) heuristics\n\n"
            "You can ask for detailed notes by typing 'ml notes', 'python notes', or 'ai notes', or ask any specific technical question!"
        )
        db = get_db()
        db.add_conversation(
            session_id=request.session_id,
            role="student",
            query=cleaned_query,
            answer=syllabus_reply,
            confidence=1.0,
            tool_used="Syllabus Overview"
        )
        return AskResponse(
            question=request.query,
            answer=syllabus_reply,
            found_in_notes=True,
            confidence=1.0,
            citations=[],
            retrieved_count=0
        )

    # 3. Dynamic Intent Routing: Support Quiz or Summary queries requested directly in chat
    try:
        from src.agents.intent_detector import IntentDetector
        detector = IntentDetector()
        intent_info = detector.detect_intent(cleaned_query)
        intent = intent_info.get("intent")
        confidence = intent_info.get("confidence", 0.0)

        if intent == "QUIZ" and confidence >= 0.75:
            rag = get_rag_pipeline()
            quiz_tool = QuizTool(vector_store=rag.vector_store)
            params = intent_info.get("parameters", {})
            quiz_res = quiz_tool.execute(topic=params.get("topic"), num_questions=params.get("num_questions", 3))
            if quiz_res.get("success") and quiz_res.get("questions"):
                questions = quiz_res["questions"]
                topic_str = params.get("topic") or "Study Notes"
                lines = [f"Here are {len(questions)} Practice Questions on '{topic_str}':\n"]
                for i, q in enumerate(questions, 1):
                    lines.append(f"Q{i}. {q['question']}")
                    for opt in q["options"]:
                        lines.append(f"   {opt}")
                    lines.append(f"   Correct Answer: {q['correct_option']}")
                    lines.append(f"   Explanation: {q['explanation']}\n")
                quiz_reply = "\n".join(lines)
                db = get_db()
                db.add_conversation(
                    session_id=request.session_id,
                    role="student",
                    query=cleaned_query,
                    answer=quiz_reply,
                    confidence=1.0,
                    tool_used="Quiz Generator"
                )
                return AskResponse(
                    question=request.query,
                    answer=quiz_reply,
                    found_in_notes=True,
                    confidence=1.0,
                    citations=[],
                    retrieved_count=0
                )

        if intent == "SUMMARY" and confidence >= 0.75:
            rag = get_rag_pipeline()
            summary_tool = SummaryTool(vector_store=rag.vector_store)
            params = intent_info.get("parameters", {})
            sum_res = summary_tool.execute(document_name=params.get("document_name"), max_points=params.get("max_points", 4))
            if sum_res.get("success") and sum_res.get("key_takeaways"):
                points = "\n".join(f"• {pt}" for pt in sum_res["key_takeaways"])
                sum_reply = f"Executive Summary of Study Notes:\n\n{sum_res.get('executive_summary', '')}\n\nKey Takeaways:\n{points}"
                db = get_db()
                db.add_conversation(
                    session_id=request.session_id,
                    role="student",
                    query=cleaned_query,
                    answer=sum_reply,
                    confidence=1.0,
                    tool_used="Summary Tool"
                )
                return AskResponse(
                    question=request.query,
                    answer=sum_reply,
                    found_in_notes=True,
                    confidence=1.0,
                    citations=[],
                    retrieved_count=0
                )
    except Exception as e:
        # Gracefully fall through to standard RAG on any agent routing exception
        pass

    try:
        # Multi-turn context resolution: get previous query from session if not passed
        prev_q = request.previous_query
        db = get_db()
        if not prev_q:
            last_convs = db.get_conversations(session_id=request.session_id, limit=1)
            if last_convs:
                prev_q = last_convs[0].get("query")

        rag = get_rag_pipeline()
        res = rag.query(cleaned_query, previous_query=prev_q)

        # Log conversation to database
        db.add_conversation(
            session_id=request.session_id,
            role="student",
            query=cleaned_query,
            answer=res["answer"],
            confidence=res["confidence"],
            tool_used="RAG Q&A"
        )

        return AskResponse(**res)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query execution failed: {str(e)}"
        )

class AskWebFallbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(default="default_session")

@app.post("/ask-web", tags=["Module 3: RAG"])
def ask_web_fallback(request: AskWebFallbackRequest) -> Dict[str, Any]:
    """Fallback to Web Search / External AI when a topic is not in uploaded notes."""
    try:
        from src.rag.web_search import WebSearchFallback
        cleaned_query = verify_and_sanitize_input(request.query)
        result = WebSearchFallback.search_and_answer(cleaned_query)

        db = get_db()
        db.add_conversation(
            session_id=request.session_id,
            role="student",
            query=cleaned_query,
            answer=result["answer"],
            confidence=1.0,
            tool_used=f"Web Fallback ({result['provider']})"
        )

        return {
            "question": request.query,
            "answer": result["answer"],
            "found_in_notes": False,
            "is_web": True,
            "provider": result["provider"],
            "confidence": 1.0,
            "citations": [
                {
                    "document_name": f"Global Web & AI Knowledge ({result['provider']})",
                    "page_number": 1,
                    "relevance_score": 1.0,
                    "snippet": "Answer synthesized via External AI / Web knowledge fallback."
                }
            ],
            "retrieved_count": 1
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"External Web fallback failed: {str(e)}"
        )

class SaveFlashcardPayload(BaseModel):
    front: str = Field(..., min_length=2)
    back: str = Field(..., min_length=2)
    topic: Optional[str] = "Saved Recall"
    source: Optional[str] = "Chat"

@app.get("/flashcards", tags=["Study Tools"])
def get_study_flashcards(topic: Optional[str] = None, count: Optional[int] = None, limit: int = 25) -> List[Dict[str, Any]]:
    """Generates interactive revision flashcards from uploaded notes and saved recall bookmarks."""
    try:
        actual_limit = count if count is not None else limit
        db = get_db()
        saved_cards = db.get_saved_flashcards()

        # If user specifically asks for 'Saved' or 'Saved Recall'
        if topic and topic.strip().lower() in ["saved", "saved recall", "favorites", "bookmarks"]:
            return saved_cards[:actual_limit]

        from src.agents.tools.flashcard_tool import FlashcardTool
        rag = get_rag_pipeline()
        tool = FlashcardTool(vector_store=rag.vector_store)
        vector_cards = tool.generate_flashcards(topic=topic, limit=actual_limit)

        # Merge saved recall cards with generated cards
        all_cards = []
        if not topic or topic.strip().lower() in ["all", ""]:
            all_cards.extend(saved_cards)
        elif topic:
            t_lower = topic.strip().lower()
            matching_saved = [c for c in saved_cards if t_lower in (c.get("topic") or "").lower() or t_lower in c.get("front", "").lower()]
            all_cards.extend(matching_saved)

        all_cards.extend(vector_cards)
        return all_cards[:actual_limit]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Flashcard generation failed: {str(e)}"
        )

@app.post("/flashcards/saved", tags=["Study Tools"])
def save_user_flashcard(payload: SaveFlashcardPayload) -> Dict[str, Any]:
    """Saves a question-answer pair from Chat or Web into 3D Flashcards for Active Recall."""
    try:
        db = get_db()
        card_id = db.save_flashcard(
            front=payload.front,
            back=payload.back,
            topic=payload.topic or "Saved Recall",
            source=payload.source or "Chat"
        )
        return {"success": True, "id": card_id, "message": "Question saved to Active Recall 3D Flashcards."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save flashcard: {str(e)}"
        )

@app.delete("/flashcards/saved/{card_id}", tags=["Study Tools"])
def delete_user_flashcard(card_id: int) -> Dict[str, Any]:
    """Deletes a saved recall flashcard by ID."""
    try:
        db = get_db()
        deleted = db.delete_saved_flashcard(card_id)
        return {"success": deleted, "message": "Flashcard removed from Saved Recall."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete flashcard: {str(e)}"
        )

@app.get("/recommendations", tags=["Dashboard"])
def get_adaptive_recommendations() -> Dict[str, Any]:
    """Returns weak topic detections and personalized learning recommendations."""
    db = get_db()
    return db.get_adaptive_recommendations()

@app.get("/history", tags=["Dashboard"])
def get_chat_history(session_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve conversation history from SQLite."""
    db = get_db()
    return db.get_conversations(session_id=session_id, limit=limit)

@app.delete("/history", tags=["Dashboard"])
def clear_chat_history(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Clear conversation history from SQLite."""
    db = get_db()
    db.clear_conversations(session_id=session_id)
    return {"success": True, "message": "Conversation history cleared successfully."}

@app.post("/summarize", response_model=SummarizeResponse, tags=["Module 4: AI Agents"])
def summarize_notes(request: SummarizeRequest) -> SummarizeResponse:
    """Summarize study notes using the Summary Tool."""
    try:
        rag = get_rag_pipeline()
        summary_tool = SummaryTool(vector_store=rag.vector_store)
        res = summary_tool.execute(
            document_name=request.document_name,
            max_points=request.max_points
        )
        return SummarizeResponse(**res)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summary Tool execution failed: {str(e)}"
        )

@app.post("/quiz", response_model=QuizResponse, tags=["Module 4: AI Agents"])
def generate_quiz(request: QuizRequest) -> QuizResponse:
    """Generate multiple-choice practice questions from notes using the Quiz Tool."""
    try:
        rag = get_rag_pipeline()
        quiz_tool = QuizTool(vector_store=rag.vector_store)
        res = quiz_tool.execute(
            topic=request.topic,
            num_questions=request.num_questions
        )
        return QuizResponse(**res)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz Tool execution failed: {str(e)}"
        )

@app.post("/quiz/submit", response_model=QuizSubmitResponse, tags=["Module 4: AI Agents"])
def submit_quiz_score(request: QuizSubmitRequest) -> QuizSubmitResponse:
    """Persist student quiz results and compute percentage."""
    db = get_db()
    db.save_quiz_result(
        topic=request.topic,
        score=request.score,
        total_questions=request.total_questions
    )
    pct = round((request.score / max(1, request.total_questions)) * 100, 1)
    return QuizSubmitResponse(
        success=True,
        message="Quiz score recorded successfully.",
        score=request.score,
        total_questions=request.total_questions,
        percentage=pct
    )

@app.post("/agent", response_model=AgentResponse, tags=["Module 4: AI Agents"])
def run_agent(request: AgentRequest) -> AgentResponse:
    """Autonomous tool selection (Answer, Summary, Quiz) via AI Agent Orchestrator."""
    cleaned_prompt = verify_and_sanitize_input(request.prompt)
    try:
        orchestrator = get_agent_orchestrator()
        res = orchestrator.run(cleaned_prompt)

        # Log agent interaction to SQLite
        db = get_db()
        db.add_conversation(
            session_id=request.session_id,
            role="student",
            query=cleaned_prompt,
            answer=res.get("agent_response", ""),
            confidence=res.get("confidence", 0.0),
            tool_used=res.get("tool_used", "AI Agent")
        )

        return AgentResponse(**res)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow encountered an error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
