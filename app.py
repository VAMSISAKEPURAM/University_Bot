"""
SV University — Document Intelligence Portal
Streamlit UI powered by FAISS + HuggingFace Embeddings + Groq LLaMA

Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile

import streamlit as st

# ── Anchor CWD to the project root ────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)

# ── Make src sub-packages importable ──────────────────────────────────────────
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from utils.helpers import load_config, setup_logging
from ingestion.loader import load_pdf
from chunking.chunkers import make_chunks
from embeddings.embedder import make_embeddings
from vector_db.vectorstores import vector_stores
from retrieval.retriever import Retriever, load_retriever
from llm.llm_clients import GroqLLMClient
from prompts.prompt_templets import RAG_PROMPT

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SV University — Document Intelligence Portal",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background  — deep navy/maroon university palette ── */
.stApp {
    background: linear-gradient(160deg, #0a0f2e 0%, #1a0a2e 40%, #0d1a3a 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,15,46,0.97) 0%, rgba(26,10,46,0.97) 100%) !important;
    border-right: 1px solid rgba(212,175,55,0.25) !important;
}
[data-testid="stSidebar"] * {
    color: #e8dfc8 !important;
}

/* ── Gold accent divider ── */
.gold-rule {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #d4af37, transparent);
    margin: 0.8rem 0;
}

/* ── Top banner / hero ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(212,175,55,0.12) 0%, rgba(10,30,80,0.6) 60%);
    border: 1px solid rgba(212,175,55,0.3);
    border-radius: 1.2rem;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    backdrop-filter: blur(12px);
}
.hero-crest {
    font-size: 3.8rem;
    flex-shrink: 0;
    filter: drop-shadow(0 0 14px rgba(212,175,55,0.5));
}
.hero-text-block {}
.hero-university {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f5e27a, #d4af37, #c8952a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    letter-spacing: 0.02em;
}
.hero-portal {
    font-size: 1rem;
    color: rgba(232,223,200,0.75);
    font-weight: 400;
    margin-top: 0.25rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.hero-tagline {
    font-size: 0.82rem;
    color: rgba(212,175,55,0.65);
    margin-top: 0.15rem;
    font-style: italic;
}

/* ── Section title ── */
.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #d4af37;
    margin-bottom: 0.5rem;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-ready    { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.35); }
.badge-notready { background: rgba(251,113,133,0.12); color: #fb7185; border: 1px solid rgba(251,113,133,0.35); }

/* ── Doc-type pill tags ── */
.tag-row { display:flex; flex-wrap:wrap; gap:0.4rem; margin:0.5rem 0; }
.tag {
    background: rgba(212,175,55,0.1);
    border: 1px solid rgba(212,175,55,0.3);
    color: #f5e27a;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    letter-spacing:0.04em;
}

/* ── Chat bubbles ── */
.chat-scroll {
    max-height: 52vh;
    overflow-y: auto;
    padding-right: 0.4rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(212,175,55,0.35) transparent;
}
.chat-scroll::-webkit-scrollbar { width: 5px; }
.chat-scroll::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.35); border-radius: 99px; }

.chat-wrap { display:flex; flex-direction:column; gap:0.9rem; padding:0.4rem 0; }

.msg-user { display:flex; justify-content:flex-end; align-items:flex-end; gap:0.55rem; }
.msg-bot  { display:flex; justify-content:flex-start; align-items:flex-end; gap:0.55rem; }

.bubble {
    max-width: 74%;
    padding: 0.8rem 1.15rem;
    border-radius: 1.1rem;
    font-size: 0.93rem;
    line-height: 1.65;
    animation: popIn 0.25s ease;
}
.bubble-user {
    background: linear-gradient(135deg, #7c2d12, #9a1c1c);
    color: #fef3c7;
    border-bottom-right-radius: 0.2rem;
    box-shadow: 0 4px 18px rgba(154,28,28,0.35);
}
.bubble-bot {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(212,175,55,0.2);
    color: #e8dfc8;
    border-bottom-left-radius: 0.2rem;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}

.avatar {
    width: 2rem; height: 2rem;
    border-radius: 50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1rem; flex-shrink:0;
}
.avatar-user { background: linear-gradient(135deg,#7c2d12,#9a1c1c); }
.avatar-bot  { background: linear-gradient(135deg,#1e3a5f,#d4af37); }

@keyframes popIn {
    from { opacity:0; transform:translateY(6px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── Context card ── */
.ctx-card {
    background: rgba(212,175,55,0.04);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 0.75rem;
    padding: 0.9rem 1rem;
    color: #c8b98a;
    font-size: 0.82rem;
    line-height: 1.7;
    max-height: 220px;
    overflow-y: auto;
    font-family: 'Inter', monospace;
}

/* ── Stat cards ── */
.stat-row { display:flex; gap:0.6rem; margin:0.5rem 0; }
.stat-card {
    flex:1;
    background: rgba(212,175,55,0.06);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 0.7rem;
    padding: 0.6rem 0.8rem;
    text-align:center;
}
.stat-num { font-size:1.4rem; font-weight:700; color:#d4af37; line-height:1.1; }
.stat-lbl { font-size:0.68rem; color:rgba(200,185,138,0.7); text-transform:uppercase; letter-spacing:0.06em; }

/* ── Empty state ── */
.empty-state {
    text-align:center; padding:3.5rem 1rem;
    color:rgba(200,185,138,0.35);
}
.empty-icon { font-size:3.2rem; }
.empty-msg  { font-size:1.05rem; margin-top:0.5rem; }
.empty-hint { font-size:0.82rem; margin-top:0.25rem; opacity:0.8; }

/* ── Streamlit widget overrides ── */
.stChatInputContainer textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(212,175,55,0.3) !important;
    border-radius: 0.75rem !important;
    color: #e8dfc8 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c2d12, #9a1c1c) !important;
    color: #fef3c7 !important;
    border: none !important;
    border-radius: 0.55rem !important;
    font-weight: 600 !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(154,28,28,0.45) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(212,175,55,0.04) !important;
    border: 1px dashed rgba(212,175,55,0.35) !important;
    border-radius: 0.8rem !important;
}
hr { border-color: rgba(212,175,55,0.15) !important; }
.stSpinner > div { border-top-color: #d4af37 !important; }
.stAlert { border-radius: 0.75rem !important; }

/* ── Info box ── */
.info-box {
    background: rgba(212,175,55,0.07);
    border-left: 3px solid #d4af37;
    border-radius: 0 0.6rem 0.6rem 0;
    padding: 0.7rem 1rem;
    color: #e8dfc8;
    font-size: 0.87rem;
    line-height: 1.6;
    margin:0.5rem 0;
}

/* ── Footer ── */
.sv-footer {
    text-align:center;
    padding: 1rem 0 0.5rem;
    color: rgba(212,175,55,0.35);
    font-size: 0.72rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "retriever": None,
        "llm_client": None,
        "rag_ready": False,
        "config": None,
        "doc_name": None,
        "chunk_count": 0,
        "department": "General",
        "doc_type": "Document",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Load config once ───────────────────────────────────────────────────────────
if st.session_state.config is None:
    setup_logging()
    st.session_state.config = load_config()

config = st.session_state.config

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # University crest in sidebar
    st.markdown("""
    <div style="text-align:center; padding:0.8rem 0 0.3rem;">
        <div style="font-size:2.8rem; filter:drop-shadow(0 0 10px rgba(212,175,55,0.5));">🎓</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.05rem; font-weight:700;
                    background:linear-gradient(90deg,#f5e27a,#d4af37);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; letter-spacing:0.03em;">
            SV University
        </div>
        <div style="font-size:0.65rem; color:rgba(212,175,55,0.5); letter-spacing:0.1em; text-transform:uppercase; margin-top:0.1rem;">
            Tirupati · Est. 1954
        </div>
    </div>
    <div class="gold-rule"></div>
    """, unsafe_allow_html=True)

    # Status
    if st.session_state.rag_ready:
        st.markdown('<span class="badge badge-ready">✅ System Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-notready">⚠️ No Document Loaded</span>', unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # ── Department & doc type ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">📚 Document Context</div>', unsafe_allow_html=True)

    department = st.selectbox(
        "Department / College",
        [
            "General",
            "College of Engineering & Technology",
            "College of Sciences",
            "College of Arts & Humanities",
            "College of Commerce & Management",
            "College of Education",
            "College of Law",
            "College of Pharmacy",
            "College of Agriculture",
            "SVU Oriental Research Institute",
            "Distance Education Centre",
        ],
        index=0,
        help="Select the department this document belongs to.",
    )
    st.session_state.department = department

    doc_type = st.selectbox(
        "Document Type",
        [
            "Academic Circular",
            "Syllabus / Curriculum",
            "Examination Notification",
            "Research Paper",
            "Administrative Notice",
            "Fee Structure",
            "Admission Guidelines",
            "Faculty Information",
            "University Regulations",
            "Project / Thesis",
            "Other",
        ],
        index=0,
    )
    st.session_state.doc_type = doc_type

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # ── Upload & Ingest ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📄 Upload College Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload any official SV University document (circular, syllabus, notice, etc.)",
        label_visibility="collapsed",
    )

    if uploaded_file and st.button("🚀 Process Document", use_container_width=True):
        with st.spinner("Processing document — chunking, embedding, indexing…"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                documents = load_pdf(tmp_path)

                chunk_cfg = config.get("chunking", {})
                chunks = make_chunks(
                    documents,
                    chunk_size=chunk_cfg.get("chunk_size", 500),
                    chunk_overlap=chunk_cfg.get("chunk_overlap", 50),
                )

                embed_cfg = config.get("embeddings", {})
                embedding_model = make_embeddings(
                    model_name=embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
                )

                vectordb_cfg = config.get("vectordb", {})
                index_path = vectordb_cfg.get("persist_directory", "faiss_index")

                vector_store = vector_stores(chunks, embedding_model, save_path=index_path)
                st.session_state.retriever = Retriever(vector_store)

                llm_cfg = config.get("llm", {})
                st.session_state.llm_client = GroqLLMClient(
                    model_name=llm_cfg.get("model_name", "llama-3.1-8b-instant")
                )

                st.session_state.rag_ready = True
                st.session_state.messages = []
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                os.unlink(tmp_path)

                st.success(f"✅ **{uploaded_file.name}** processed successfully!\n\n`{len(chunks)}` chunks indexed.")
                st.rerun()

            except Exception as exc:
                st.error(f"❌ Processing failed: {exc}")

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # ── Load existing index ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🗄️ Load Saved Index</div>', unsafe_allow_html=True)
    if st.button("🔄 Load Existing FAISS Index", use_container_width=True):
        with st.spinner("Loading index from disk…"):
            try:
                vectordb_cfg = config.get("vectordb", {})
                index_path = vectordb_cfg.get("persist_directory", "faiss_index")
                st.session_state.retriever = load_retriever(index_path)

                llm_cfg = config.get("llm", {})
                st.session_state.llm_client = GroqLLMClient(
                    model_name=llm_cfg.get("model_name", "llama-3.1-8b-instant")
                )

                st.session_state.rag_ready = True
                st.success("✅ Index loaded!")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ {exc}")

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # ── Document stats ─────────────────────────────────────────────────────────
    if st.session_state.rag_ready:
        st.markdown('<div class="section-title">📊 Session Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-num">{st.session_state.chunk_count or "—"}</div>
                <div class="stat-lbl">Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{len(st.session_state.messages) // 2}</div>
                <div class="stat-lbl">Queries</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.doc_name:
            st.markdown(f"""
            <div class="info-box">
                📄 <strong>{st.session_state.doc_name}</strong><br>
                🏛️ {st.session_state.department}<br>
                🏷️ {st.session_state.doc_type}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ── Model info ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🤖 AI Engine</div>', unsafe_allow_html=True)
    llm_cfg   = config.get("llm", {})
    embed_cfg = config.get("embeddings", {})
    st.markdown(f"""
    <div style="font-size:0.75rem; color:rgba(200,185,138,0.65); line-height:2;">
        <b>LLM</b>: {llm_cfg.get('model_name','—')}<br>
        <b>Embeddings</b>: all-MiniLM-L6-v2<br>
        <b>Vector DB</b>: FAISS (local)
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-crest">🎓</div>
    <div class="hero-text-block">
        <div class="hero-university">Sri Venkateswara University</div>
        <div class="hero-portal">Document Intelligence Portal</div>
        <div class="hero-tagline">AI-Powered Academic Document Analysis &amp; Q&amp;A · Tirupati, Andhra Pradesh</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick suggestion pills ─────────────────────────────────────────────────────
st.markdown("""
<div class="section-title">💡 Common Queries</div>
<div class="tag-row">
    <span class="tag">📋 Summarise this document</span>
    <span class="tag">📅 Key dates &amp; deadlines</span>
    <span class="tag">📜 Eligibility criteria</span>
    <span class="tag">💰 Fee details</span>
    <span class="tag">📘 Course structure</span>
    <span class="tag">📞 Contact &amp; helpline</span>
    <span class="tag">🗓️ Exam schedule</span>
    <span class="tag">📝 Admission procedure</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Two-column layout: Chat | Context ─────────────────────────────────────────
chat_col, ctx_col = st.columns([3, 1], gap="medium")

with chat_col:
    st.markdown('<div class="section-title">💬 Document Q&A Conversation</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <div class="empty-msg">No conversation yet</div>
            <div class="empty-hint">
                Upload an SV University document using the sidebar,<br>
                then ask anything — summaries, deadlines, eligibility, fees…
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chat_html = '<div class="chat-scroll"><div class="chat-wrap">'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="msg-user">
                    <div class="bubble bubble-user">{msg['content']}</div>
                    <div class="avatar avatar-user">🧑</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="msg-bot">
                    <div class="avatar avatar-bot">🎓</div>
                    <div class="bubble bubble-bot">{msg['content']}</div>
                </div>"""
        chat_html += "</div></div>"
        st.markdown(chat_html, unsafe_allow_html=True)

with ctx_col:
    st.markdown('<div class="section-title">📖 Retrieved Context</div>', unsafe_allow_html=True)

    last_ctx = None
    for msg in reversed(st.session_state.messages):
        if msg.get("context"):
            last_ctx = msg["context"]
            break

    if last_ctx:
        st.markdown(
            f'<div class="ctx-card">{last_ctx.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ctx-card" style="color:rgba(200,185,138,0.3); text-align:center; padding:2.5rem 0.5rem;">'
            '📄<br><br>Retrieved document<br>excerpts will appear here</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Chat input ────────────────────────────────────────────────────────────────
if not st.session_state.rag_ready:
    st.markdown("""
    <div class="info-box">
        ⬅️ &nbsp;<strong>Upload an SV University document</strong> from the sidebar to begin.<br>
        Supported documents: circulars, syllabi, examination notices, fee structures,
        admission guidelines, research papers, and more.
    </div>
    """, unsafe_allow_html=True)
else:
    user_query = st.chat_input(
        f"Ask about this {st.session_state.doc_type.lower()} from {st.session_state.department}…"
    )

    if user_query and user_query.strip():
        st.session_state.messages.append(
            {"role": "user", "content": user_query.strip(), "context": None}
        )

        with st.spinner("🔍 Searching document and generating answer…"):
            try:
                context = st.session_state.retriever.retrieve(user_query.strip())

                if not context:
                    answer = (
                        "I couldn't find relevant information in the uploaded document. "
                        "Please try rephrasing your question or upload a different document."
                    )
                    context = ""
                else:
                    prompt = RAG_PROMPT.format(context=context, question=user_query.strip())
                    answer = st.session_state.llm_client.generate(prompt)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "context": context}
                )

            except Exception as exc:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ Error: {exc}", "context": None}
                )

        st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sv-footer">
    🎓 Sri Venkateswara University · Tirupati, Andhra Pradesh — 517502<br>
    Document Intelligence Portal · Powered by FAISS · HuggingFace Embeddings · Groq LLaMA 3.1
</div>
""", unsafe_allow_html=True)
