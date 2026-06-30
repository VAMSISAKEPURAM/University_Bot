# RAG Chatbot — Code Review & Corrections

## Summary

Your project has a solid folder structure but several critical issues prevent it from running:
- Two core files are **completely empty** (`main.py`, `llm_clients.py`, `prompt_templets.py`)
- The retriever is **hardcoded** instead of accepting dynamic queries
- The embedder returns raw vectors but **doesn't return the model object** needed by FAISS
- The vector store **doesn't return** the created store (no `return` statement)
- An **exposed HuggingFace API token** in `.env` (you should regenerate it)
- Import in `routes.py` references a module name that doesn't match the actual filename
- The `config.yaml` references `chroma_db` but the code uses **FAISS**
- `requirements.txt` references packages that aren't actually used in the code

---

## Issues Found & Fixes Applied

### 🔴 CRITICAL — Empty Files

#### `src/main.py` — was completely empty
This is the entry point. Without it, the application cannot run at all.

**Why it needs an entry point**: `main.py` must wire all modules together — load config, ingest PDFs, initialize the vector store + retriever + LLM client, and start the FastAPI server.

---

#### `src/llm/llm_clients.py` — was completely empty
The `routes.py` calls `_llm_client.generate(prompt)`, so an LLM client class must exist.

**Why**: Since you are using a HuggingFace token (`.env`), the correct approach is to use `HuggingFaceHub` or the `InferenceClient` from the `huggingface_hub` library.

---

#### `src/prompts/prompt_templets.py` — was completely empty (also misspelled)
The `routes.py` imports `from prompts import prompt_templates` (note: `templates` not `templets`), but the file is named `prompt_templets.py`. This causes an **ImportError** at runtime.

**Two problems**:
1. File is empty — no `RAG_PROMPT` defined
2. Import in `routes.py` says `prompt_templates` but file is `prompt_templets` — mismatch

---

### 🔴 CRITICAL — Wrong Import in `routes.py`

```python
# BEFORE (broken — module name mismatch)
from prompts import prompt_templates

# AFTER (fixed to match actual filename)
from prompts.prompt_templets import RAG_PROMPT
```

---

### 🟠 HIGH — `retriever.py` has a hardcoded query

```python
# BEFORE (broken — query is hardcoded, not usable as a real retriever)
def retrive_chunks(db_path):
    vector_store = db_path
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    query = "What is software engineering?"   # ← hardcoded!
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs
```

**Why it's wrong**: A retriever in a chatbot must accept the user's actual question, not a hardcoded string.  
Also: the function name `retrive_chunks` has a typo (`retrive` instead of `retrieve`).  
Also: `routes.py` calls `_retriever.retrieve(request.question)` — it expects a **class instance with a `.retrieve()` method**, not a bare function.

---

### 🟠 HIGH — `embedder.py` returns raw vectors instead of the embedding model

```python
# BEFORE (wrong — returns a list of float arrays, useless for FAISS)
def make_embeddings(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    document_embeddings = embeddings.embed_documents([doc.page_content for doc in documents])
    return document_embeddings
```

**Why it's wrong**: `FAISS.from_documents()` in `vectorstores.py` needs the **embedding model object** itself, not pre-computed vectors. The model is passed so FAISS can embed both the documents AND future queries.

---

### 🟠 HIGH — `vectorstores.py` missing `return` statement

```python
# BEFORE (broken — vector store is created but never returned)
def vector_stores(splits, embeddings):
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
    vector_store.save_local("faiss_index")
    # ← nothing returned! caller gets None
```

**Why it's wrong**: The calling code needs the `vector_store` object to create a retriever. Without `return`, the function produces `None`.

---

### 🟠 HIGH — `config.yaml` vs code mismatch

`config.yaml` says:
```yaml
vectordb:
  persist_directory: "./chroma_db"   # ChromaDB path
  collection_name: "pdf_knowledge"
llm:
  provider: "transformers"
  model_name: "google/flan-t5-base"
```

But the code uses **FAISS** (not ChromaDB) and the `.env` contains a **HuggingFace API token** suggesting `HuggingFaceHub` inference, not local `transformers`. Config and code need to be in sync.

---

### 🟡 MEDIUM — `.env` has an exposed API token (security risk)

The HuggingFace token in `.env` is visible in this file. While `.gitignore` correctly excludes `.env`, **you should regenerate this token immediately** at https://huggingface.co/settings/tokens since it was exposed.

The `.env` key should be named consistently:
```
# Standard naming convention
HUGGINGFACEHUB_API_TOKEN=hf_...
```

---

### 🟡 MEDIUM — `requirements.txt` has version issues

| Package | Issue |
|---|---|
| `torch==2.12.1` | This version doesn't exist. Latest stable is ~2.3.x. Remove pinned version or use a valid one. |
| `langchain-text-splitters==0.0.1` | Very old version. Use `>=0.2.0` |
| `chromadb==0.4.22` | You're not using ChromaDB in the code — FAISS is used instead. Remove or replace with `faiss-cpu`. |
| `langchain_huggingface` | Used in `embedder.py` but **not listed** in `requirements.txt` |
| `huggingface_hub` | Needed for `InferenceClient` but **not listed** |

---

### 🟡 MEDIUM — `chunkers.py` ignores `config.yaml` values

```python
# BEFORE (hardcoded — ignores config.yaml settings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

`config.yaml` defines `chunk_size: 500` and `chunk_overlap: 50`. The chunker should read these from config so they can be adjusted without touching code.

---

### 🟢 LOW — Typo in filename: `prompt_templets.py`

Should be `prompt_templates.py`. While renaming is optional (you can keep it as-is and just fix the import), consistent naming matters for readability.

---

## Architecture Diagram (correct flow)

```
PDF File
   │
   ▼
loader.py  ──► PyPDFLoader → list[Document]
   │
   ▼
chunkers.py ──► RecursiveCharacterTextSplitter → list[Document chunks]
   │
   ▼
embedder.py ──► HuggingFaceEmbeddings MODEL OBJECT (not vectors!)
   │
   ▼
vectorstores.py ──► FAISS.from_documents(chunks, embedding_model)
                    → saves to disk + RETURNS vector_store object
   │
   ▼
retriever.py ──► Retriever class with .retrieve(query) method
   │
   ▼
routes.py ──► /ask endpoint
              1. retriever.retrieve(question) → context chunks
              2. prompt_templets.RAG_PROMPT.format(context, question)
              3. llm_client.generate(prompt) → answer
   │
   ▼
main.py ──► Wires everything, starts uvicorn server
```
