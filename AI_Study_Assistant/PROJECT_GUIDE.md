# AI Study Assistant - Complete Tech Stack & Viva Explanation Guide

Yeh guide is project me use hui har ek technology, library, math formula aur concept ko simple Hinglish me explain karti hai taaki aap viva ya presentation me bina kisi confusion ke Sir ko samjha sakein.

---

## 1. Project Ka Ek Nazar Me Overview

- **Project Name**: AI Study Assistant (Student Knowledge Portal)
- **Main Goal**: Uploaded college study notes (PDF / Text) par grounded personal AI tutor provide karna jo questions classify kare, notes se exact answer de, quiz banaye, aur viva me model ki prediction explain kare.
- **Architecture**: Single Page Application (React) + FastAPI REST Backend + PyTorch Deep Learning + HuggingFace Transformers + FAISS Vector Search + Local XAMPP MySQL Database.

---

## 2. Har Module Me Kya Use Hua Hai (Tech Stack & Viva Answers)

### Module 1: Deep Learning & Neural Networks (Question Classification)
- **Kaam**: Student ke question ko classify karna ki yeh **Python**, **AI**, ya **Machine Learning** ka sawal hai.
- **Libraries Used**:
  - `torch` (PyTorch 2.x): Neural network build aur train karne ke liye.
  - `torch.nn`: Linear layers, Batch Normalization, ReLU, Dropout define karne ke liye.
  - `scikit-learn` (`TfidfVectorizer`, `LabelEncoder`): Text ko numerical vector me convert karne aur labels encode karne ke liye.
  - `numpy`: Matrix computations aur arrays ke liye.
- **Model Architecture**:
  - Input Layer: 800 features (TF-IDF bag of n-grams).
  - Hidden Layer 1: 128 neurons + BatchNorm1d + ReLU + Dropout(0.3).
  - Hidden Layer 2: 64 neurons + BatchNorm1d + ReLU + Dropout(0.21).
  - Output Layer: 3 logits (Python, AI, ML).
  - Loss Function: CrossEntropyLoss.
  - Optimizer: AdamW.
  - Test Accuracy: 96.15%.
- **Files**:
  - Model Definition: `src/classification/model.py`
  - Training Pipeline: `src/classification/train.py`
  - Inference & XAI: `src/classification/predict.py`
- **Sir ko kya bolna hai**:
  > "Sir, humne PyTorch me ek 3-layer MLP train kiya hai. Overfitting rokne ke liye Batch Normalization aur Dropout lagaya hai, aur text representation ke liye Scikit-Learn ka TF-IDF use kiya hai."

---

### Module 2: NLP & Pretrained Transformers (Question Understanding)
- **Kaam**: Text cleaning, tokenization, transformer attention math aur dense semantic embeddings generate karna.
- **Libraries Used**:
  - `nltk`: Text normalization, regex tokenization, PorterStemmer, aur WordNetLemmatizer.
  - `sentence-transformers`: HuggingFace ka pretrained transformer model `all-MiniLM-L6-v2`.
- **Core Concepts**:
  - **Stemming vs Lemmatization**: Stemming crude word truncation karta hai ("running" -> "run"), jabki lemmatization dictionary vocabulary dekh kar linguistic root nikaalta hai ("studies" -> "study").
  - **Scaled Dot-Product Attention**: Transformer ka core formula mathematically implement kiya gaya:
    `Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V`
  - **Dense Embeddings**: Har question ka 384-dimensional dense semantic vector banta hai jo words ke meaning ko capture karta hai (sirf keyword matching nahi).
- **Files**:
  - Text Normalizer: `src/nlp/preprocessor.py`
  - Transformer & Attention Math: `src/nlp/transformer_pipeline.py`
- **Sir ko kya bolna hai**:
  > "Sir, humne NLP me NLTK se stemming/lemmatization ki hai aur HuggingFace ke all-MiniLM-L6-v2 transformer se 384-dimensional dense semantic embeddings banaye hain. Saath hi attention mechanism ka scaled dot-product formula bhi code kiya hai."

---

### Module 3: Generative AI & RAG (Retrieval-Augmented Generation)
- **Kaam**: Uploaded PDF/Text study notes ko search karke grounded answer generate karna aur hallucinations rokna.
- **Libraries Used**:
  - `faiss-cpu`: Facebook AI Similarity Search library (FlatIP index for Cosine similarity).
  - `pypdf`: PDF documents se text extract karne ke liye.
- **Core Concepts**:
  - **Sliding Window Chunking**: Notes ko 500 characters ke tukdon me toda gaya with 100 character overlap, taaki do chunks ke beech context na toote.
  - **Vector Database (FAISS)**: Har chunk ka 384-d vector FAISS FlatIP index me save hota hai.
  - **Anti-Hallucination Guardrail**: Cosine similarity score ka threshold `0.35` set kiya hai. Agar question notes ke syllabus se bahar ka hoga (similarity < 0.35), to system safely bol deta hai: `[Out of Notes Scope]`.
  - **Grounded Citations**: Har answer ke saath document ka naam aur page number attach hota hai.
  - **Context Memory (Multi-Turn)**: Pichle question ka context yaad rakh kar follow-up sawalon ka sahi answer deta hai.
- **Files**:
  - Document Reader: `src/rag/document_loader.py`
  - Chunker: `src/rag/chunker.py`
  - Vector Database: `src/rag/vector_store.py`
  - Pipeline & Context Memory: `src/rag/pipeline.py`
- **Sir ko kya bolna hai**:
  > "Sir, RAG me humne FAISS FlatIP index use kiya hai. Notes ko 500-char sliding window me chunk kiya hai. False answers rokne ke liye 0.35 cosine threshold ka anti-hallucination guardrail lagaya hai aur har response me exact document name aur page number cite hota hai."

---

### Module 4: AI Agents & Automation (Autonomous Tool Routing)
- **Kaam**: User ke natural language prompt se automatic intent pehchan kar sahi tool chalana.
- **Core Components**:
  - `IntentDetector`: Rule-based patterns aur semantic keyword weights se user intent classify karta hai (`ANSWER`, `SUMMARY`, `QUIZ`, `FLASHCARD`).
  - `AgentOrchestrator`: Master agent jo intent ke according tool choose karta hai.
  - **Tools**:
    - `AnswerTool`: RAG Pipeline ko execute karta hai.
    - `SummaryTool`: Notes ka Executive Summary banata hai.
    - `QuizTool`: Notes se Multiple Choice Questions (MCQs) generate karta hai.
    - `FlashcardTool`: Notes se 3D interactive flashcards extract karta hai.
- **Files**:
  - Intent Detector: `src/agents/intent_detector.py`
  - Orchestrator: `src/agents/orchestrator.py`
  - Tools: `src/agents/tools/summary_tool.py`, `src/agents/tools/quiz_tool.py`, `src/agents/tools/flashcard_tool.py`
- **Sir ko kya bolna hai**:
  > "Sir, traditional bot me user ko buttons dabane padte hain. Hamara AI Agent autonomous router hai jo student ka natural language prompt samajh kar khud decide karta hai ki kab Quiz banana hai, kab Summary deni hai, aur kab Answer karna hai."

---

### Module 5: Deployment, Security Guardrails & Database
- **Kaam**: API server chalana, inputs validate karna, sensitive data protect karna aur user history store karna.
- **Libraries Used**:
  - `fastapi`: High-performance asynchronous REST API framework.
  - `uvicorn`: ASGI server jo application ko host karta hai.
  - `pydantic`: Incoming requests aur outgoing responses ka strict schema validation.
  - `pymysql` + `cryptography`: Python se XAMPP MySQL database connect karne ke liye.
- **Security Guardrails**:
  - **PII Masking Filter**: Email (`[REDACTED_EMAIL]`), Phone number (`[REDACTED_PHONE]`), API keys, credit cards ko detect karke mask karta hai.
  - **Prompt Injection Defense**: System prompt bypass attacks (jaise `"ignore previous instructions"`) ko block karta hai.
  - **Rate Limiting**: Sliding window rate limiter (max 30 requests/minute per client).
- **Database Architecture**:
  - Primary Engine: Local XAMPP MySQL Database (`study_assistant` on `127.0.0.1:3306`).
  - Tables: `documents`, `conversations`, `quizzes` (51 active records).
  - Fallback: Local SQLite file (`study_assistant.db`) agar MySQL band ho to system crash nahi hota.
- **Files**:
  - FastAPI App: `api/main.py`
  - Security Guardrails: `src/guardrails/security.py`
  - Database Manager: `src/database/db.py`
- **Sir ko kya bolna hai**:
  > "Sir, backend FastAPI ASGI server par run ho raha hai with Pydantic validation. Security me humne PII redaction filter, Prompt Injection guardrail aur Rate Limiter lagaya hai. Data persistence ke liye XAMPP MySQL database connected hai with automatic SQLite fallback."

---

### Special Standout Features (Extra Marks & Viva Impact)

1. **Explainable AI (XAI)**:
   - Method: `Input * Gradient Saliency Attribution`.
   - Kaam: Neural network output ka backward pass karke exact word influence percentage nikaalta hai (jaise: `'gradient descent': 35.4%`).
   - File: `src/classification/predict.py`.

2. **Interactive 3D Flip Flashcards**:
   - Method: CSS3 3D Perspective + Active Recall Card Extraction.
   - Kaam: Student revision ke liye 3D cards flip karke concepts aur key takeaways dekh sakta hai.

3. **Weak Topic Detection & Smart Tutor Recommendations**:
   - Method: MySQL Quiz Analytics Engine.
   - Kaam: Student ke 35 quizzes ka topic-wise score calculate karke weak subject (Python 88% vs ML 99%) identify karta hai aur tailored study guidance deta hai.

4. **Voice Input (Speech-to-Text)**:
   - Method: HTML5 Web Speech API (`webkitSpeechRecognition`).
   - Kaam: Mic dabakar bolne se speech automatically query text me convert hoti hai.

5. **Smart Follow-up Context Memory**:
   - Method: Conversational Multi-turn Entity Resolution.
   - Kaam: Pichle question ka context attach karke continuous grounded answers generate karta hai.

6. **1-Click Export Study Guide**:
   - Method: Client-side Markdown File Blob Generator.
   - Kaam: Poore chat history ko citations aur confidence scores ke saath `.md` file me download karta hai.

---

## 3. Quick Viva Q&A Cheat Sheet (Agar Sir Puchein)

| Sir Ka Sawal | Aapka 1-Line Answer |
|---|---|
| Deep learning me kya kiya? | PyTorch me 3-layer MLP neural network train kiya (96.15% accuracy) jo question ko Python/AI/ML me classify karta hai. |
| NLP me kya use kiya? | NLTK se stemming/lemmatization, PyTorch attention math, aur sentence-transformers se 384-d semantic embeddings. |
| RAG kaise chal raha hai? | FAISS FlatIP index se 500-char notes chunks search hote hain, 0.35 cosine threshold se hallucination rukti hai, aur citations aati hain. |
| AI Agent kya karta hai? | Natural language prompt se automatic intent pehchan kar Quiz Tool, Summary Tool ya Answer Tool execute karta hai. |
| Deployment & Database me kya hai? | FastAPI ASGI server, Pydantic validation, PII masking guardrail, aur XAMPP MySQL database (port 3306). |
| Extra unique kya banaya hai? | Explainable AI (XAI Saliency), 3D Flip Flashcards, Weak Topic Detection, Voice Input aur Context Memory. |

---

## 4. Run Karne Ki Command

```bash
# Terminal me project folder me:
python run.py
```
Browser me: **`http://127.0.0.1:8000/dashboard`**
Database: XAMPP phpMyAdmin -> **`study_assistant`**
