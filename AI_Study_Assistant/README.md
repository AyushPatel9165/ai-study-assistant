# AI Study Assistant

An end-to-end, production-grade AI-powered study assistant built with PyTorch, Hugging Face Transformers, FAISS, and FastAPI.

## Key Features Across 5 Modules

### Module 1: Deep Learning & Neural Networks
- **Student Question Classification**: Multi-Layer Perceptron (MLP) trained on Python, Artificial Intelligence, and Machine Learning technical questions.
- **Architecture**: `QuestionClassifierMLP` (Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear -> BatchNorm1d -> ReLU -> Linear).
- **Validation Accuracy**: 96.15% test accuracy and 0.9617 weighted F1-score with zero data leakage.

### Module 2: NLP & Transformers
- **Preprocessing Pipeline**: Tokenization, lowercase normalization, punctuation removal, stopword filtering, stemming vs. lemmatization.
- **Transformer Embeddings**: Pretrained dense 384-dimensional representations via `sentence-transformers/all-MiniLM-L6-v2` with offline caching.
- **Mathematical Attention**: Full implementation of Scaled Dot-Product Self-Attention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Module 3: Generative AI & Retrieval-Augmented Generation (RAG)
- **Document Ingestion**: Parsing PDF and TXT lecture notes with character sliding window chunking (400 chars, 60 overlap).
- **FAISS Vector Index**: Semantic search via `IndexFlatIP` (cosine similarity on normalized vectors).
- **Anti-Hallucination Guardrail**: Queries with maximum similarity below 0.35 are strictly rejected to prevent hallucinations.
- **Grounded Answers with Citations**: Responses include document name, page number, and similarity score.

### Module 4: AI Agents & Autonomous Tool Selection
- **Intent Detection (`src/agents/intent_detector.py`)**: Identifies student goals (`ANSWER`, `SUMMARY`, `QUIZ`) and extracts parameters.
- **Specialized Tools**:
  - `AnswerTool`: Grounded Q&A via RAG pipeline.
  - `SummaryTool`: Synthesizes executive summaries and key bullet points.
  - `QuizTool`: Generates interactive multiple-choice questions with 4 options, correct answer, and detailed explanations.
- **Autonomous Orchestrator (`src/agents/orchestrator.py`)**: Coordinates state machine execution and produces structured traces.

### Module 5: Deployment, Security & Guardrails
- **FastAPI Endpoints**:
  - `POST /classify`: Live PyTorch neural network classification.
  - `POST /ask`: RAG-based grounded answering.
  - `POST /upload`: Upload and index PDF/TXT study notes.
  - `POST /summarize`: Automatic note summarization.
  - `POST /quiz`: Practice MCQ generation.
  - `POST /quiz/submit`: Record scores and calculate performance.
  - `POST /agent`: Autonomous agent routing.
  - `GET /metrics`: Aggregated metrics for dashboard cards.
  - `GET /documents`: Registered documents list.
  - `GET /dashboard`: Interactive Single-Page React Dashboard.
- **Security Guardrails**:
  - `PromptInjectionGuard`: Blocks adversarial overrides and DAN jailbreaks.
  - `SensitiveDataFilter`: Masks emails, phone numbers, credit cards, and API secrets.
  - `RateLimiter`: Sliding-window limiter per IP address.
- **Persistence**: SQLite database (`data/study_assistant.db`) storing documents, conversations, and quiz scorecards.

---

## Project Structure

```
AI_Study_Assistant/
    api/
        main.py
    data/
        raw/
            student_questions.csv
            sample_study_notes.txt
        study_assistant.db
    frontend/
        index.html
    models/
        question_classifier.pt
        tfidf_vectorizer.pkl
        label_encoder.pkl
        model_metadata.json
        faiss_index.bin
        faiss_chunks.json
    notebooks/
        01_module1_deep_learning_exploration.ipynb
        02_module2_nlp_transformers_exploration.ipynb
        03_module3_generative_ai_rag_exploration.ipynb
        04_module4_ai_agents_exploration.ipynb
    src/
        agents/
            intent_detector.py
            orchestrator.py
            tools/
                answer_tool.py
                summary_tool.py
                quiz_tool.py
        classification/
            dataset.py
            model.py
            predict.py
            train.py
        database/
            db.py
        guardrails/
            security.py
        nlp/
            preprocessor.py
            transformer_pipeline.py
        rag/
            chunker.py
            document_loader.py
            generator.py
            pipeline.py
            vector_store.py
    tests/
        test_environment.py
        test_classification.py
        test_nlp.py
        test_rag.py
        test_agents.py
        test_guardrails_and_db.py
        test_api_base.py
    requirements.txt
```

---

## How to Run the Application

### 1. Activate Environment & Run Tests
```powershell
cd AI_Study_Assistant
venv\Scripts\pytest tests -v
```
All unit and integration tests will execute and pass.

### 2. Start the FastAPI Backend & Dashboard
```powershell
cd AI_Study_Assistant
venv\Scripts\uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Open the Application in Your Browser
- **Interactive Engineering Dashboard**: `http://127.0.0.1:8000/dashboard`
- **Interactive Swagger API Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc API Documentation**: `http://127.0.0.1:8000/redoc`

---

## Sample API Requests

### 1. Subject Classification
```bash
curl -X POST "http://127.0.0.1:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"question": "How does backpropagation update weights in a neural network?"}'
```

### 2. Grounded RAG Question Answering
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the Global Interpreter Lock in Python?"}'
```

### 3. Autonomous AI Agent
```bash
curl -X POST "http://127.0.0.1:8000/agent" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Create 3 practice MCQs on machine learning regularization"}'
```
