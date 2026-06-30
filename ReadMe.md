# 🎓 SV University — Document Intelligence Portal

> AI-powered academic document analysis and Q&A system built for **Sri Venkateswara University, Tirupati**.  
> Upload any university document (circular, syllabus, exam notice, fee structure, etc.) and instantly query it using natural language.

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit (university-themed) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs locally) |
| **Vector Store** | FAISS (local, persisted to disk) |
| **LLM** | `llama-3.1-8b-instant` via **Groq Inference API** |
| **PDF Loader** | LangChain + PyPDF |
| **REST API** | FastAPI + Uvicorn |

---

## 📁 Project Structure

```
Rag_flow/
├── app.py                  # Streamlit UI (SV University themed)
├── config.yaml             # Central config (model names, chunk sizes, paths)
├── .env                    # API keys (GROQ_API_KEY)
├── requirements.txt
├── faiss_index/            # Persisted FAISS vector index (auto-created)
└── src/
    ├── main.py             # CLI entry point (ingest + FastAPI server)
    ├── api/
    │   └── routes.py       # FastAPI /ask and /health endpoints
    ├── ingestion/
    │   └── loader.py       # PDF loading via LangChain PyPDFLoader
    ├── chunking/
    │   └── chunkers.py     # Text splitting (RecursiveCharacterTextSplitter)
    ├── embeddings/
    │   └── embedder.py     # HuggingFace embeddings factory
    ├── vector_db/
    │   └── vectorstores.py # FAISS index builder & saver
    ├── retrieval/
    │   └── retriever.py    # FAISS retriever wrapper
    ├── llm/
    │   └── llm_clients.py  # GroqLLMClient (llama-3.1-8b-instant)
    ├── prompts/
    │   └── prompt_templets.py  # RAG prompt template
    └── utils/
        └── helpers.py      # Config loader, logging setup
```

---

## ⚙️ Setup

### 1. Clone & create virtual environment

```powershell
git clone <repo-url>
cd Rag_flow
python -m venv myvenv
myvenv\Scripts\activate        # Windows
# source myvenv/bin/activate   # macOS / Linux
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API key

Open `.env` and add your **Groq API key** (get one free at [console.groq.com/keys](https://console.groq.com/keys)):

```env
GROQ_API_KEY=gsk_your_key_here
```

> The HuggingFace embedding model runs **fully locally** — no HuggingFace token required.

### 4. (Optional) Edit `config.yaml`

```yaml
chunking:
  chunk_size: 500
  chunk_overlap: 50

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

llm:
  provider: "groq"
  model_name: "llama-3.1-8b-instant"
```

---

## 🚀 Running the App

### ▶ Streamlit UI (recommended)

```powershell
myvenv\Scripts\streamlit run app.py
```

Open **http://localhost:8501** in your browser.

**Workflow inside the UI:**
1. Select the **Department / College** and **Document Type** from the sidebar
2. Upload a PDF (circular, syllabus, notice, etc.)
3. Click **Process Document** — the system chunks, embeds, and indexes it
4. Type your question in the chat box (e.g. *"Summarise this document"*, *"What are the key deadlines?"*)
5. The AI retrieves relevant context from the document and answers via Groq LLaMA

---

### ▶ CLI — Ingest a PDF

```powershell
myvenv\Scripts\python src/main.py --ingest path\to\document.pdf
```

### ▶ CLI — Start FastAPI Server

```powershell
myvenv\Scripts\python src/main.py --serve
```

API docs available at **http://localhost:8000/docs**

**POST `/ask`**
```json
{ "question": "What are the exam dates mentioned in this document?" }
```

### ▶ Ingest + Serve in one command

```powershell
myvenv\Scripts\python src/main.py --ingest document.pdf --serve
```

---

## 🏛️ Supported Document Types

The portal is designed for SV University academic documents including:

- 📋 Academic Circulars
- 📘 Syllabi & Curriculum Documents
- 📅 Examination Notifications & Schedules
- 💰 Fee Structures
- 📝 Admission Guidelines
- 🔬 Research Papers & Theses
- 📜 University Regulations & Ordinances
- 👥 Faculty & Administrative Notices

---

## 🛑 Stopping the App

Press **`Ctrl + C`** in the terminal where the app is running.

---

## 🔧 Troubleshooting

| Issue | Fix |
|---|---|
| `GROQ_API_KEY not found` | Add your key to `.env`: `GROQ_API_KEY=gsk_...` |
| `config.yaml not found` | Always run from the project root directory |
| `FAISS index not found` | Upload and process a PDF first before loading the index |
| Streamlit static files missing | Reinstall: `pip uninstall streamlit -y` then `pip install streamlit` |

---

## 📜 License

For academic and internal use within **Sri Venkateswara University, Tirupati, Andhra Pradesh — 517502**.