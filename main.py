import pickle
from pathlib import Path
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

import streamlit as st

import streamlit as st

# ✅ ADD THIS — must be first st.* call
st.set_page_config(
    page_title="CorvitRAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"   # 👈 forces sidebar open in iframe
)



# ── Setup & Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PDF_PATH, STORE_DIR = ROOT / "asset/data/corvit.pdf", ROOT / "asset/vectorstore"
INDEX_FILE, CHUNKS_FILE = STORE_DIR / "index.faiss", STORE_DIR / "chunks.pkl"
EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP = "sentence-transformers/all-MiniLM-L6-v2", 700, 120

GROQ_MODELS = [
    "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"
]

SYSTEM_PROMPT = """You are CorvitRAG, assistant for Corvit Systems (IT training, Pakistan).
Answer ONLY from the CONTEXT below.
- If missing, say: "I could not find this in the Corvit knowledge base. Please check the official website or contact a campus."
- Do not invent prices, dates, phones, or eligibility. Prices are catalog prices — advise verification.
- Be concise. Use bullets for lists.

CONTEXT:
{context}"""

# ── Streamlit Config & Minimal Dark CSS ────────────────────────────────────
st.set_page_config(page_title="CorvitRAG", page_icon="🎓", layout="wide")
st.markdown("""<style>
:root { --bg: #080c14; --panel: #0d1521; --border: #202e40; --text: #e8edf4; --accent: #1ea7ef; }
html, body, .stApp, [data-testid="stAppViewContainer"], section[data-testid="stSidebar"] { background: var(--bg) !important; color: var(--text) !important; }
[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; height: 0 !important; }

/* Global Input & Dropdown Overrides (Fixes light/white boxes) */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div, 
div[role="listbox"], li[role="option"], [data-testid="stChatInput"] textarea {
    background-color: var(--panel) !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 8px !important;
}
li[role="option"]:hover { background-color: #152235 !important; }
.msg-user { width: fit-content; max-width: 78%; margin: 8px 0 8px auto; background: #122b43; color: #eaf4fc; border: 1px solid #1b405f; border-radius: 12px 12px 2px 12px; padding: 10px 14px; }
.msg-bot { width: fit-content; max-width: 84%; margin: 8px 0; background: var(--panel); color: #dce5ef; border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 2px 12px 12px 12px; padding: 10px 14px; }
</style>""", unsafe_allow_html=True)

# ── State & Helpers ────────────────────────────────────────────────────────
for k, v in {"connected": False, "api_key": "", "model": GROQ_MODELS[0], "messages": [], "chunks": None, "index": None,
             "embedder": None, "n_chunks": 0}.items():
    st.session_state.setdefault(k, v)


@st.cache_resource(show_spinner=False)
def get_embedder(): return SentenceTransformer(EMBED_MODEL)


def build_or_load_index(force=False):
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    if INDEX_FILE.exists() and CHUNKS_FILE.exists() and not force:
        with open(CHUNKS_FILE, "rb") as f: return faiss.read_index(str(INDEX_FILE)), pickle.load(f)
    if not PDF_PATH.exists(): raise FileNotFoundError(f"Missing PDF: {PDF_PATH}")

    text = "\n\n".join(p.extract_text() or "" for p in PdfReader(str(PDF_PATH)).pages)
    chunks = [text[i:i + CHUNK_SIZE].strip() for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP) if
              text[i:i + CHUNK_SIZE].strip()]
    embedder = get_embedder()
    vectors = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    return index, chunks


def query_rag(prompt):
    embedder, index, chunks = st.session_state.embedder, st.session_state.index, st.session_state.chunks
    q_vec = embedder.encode([prompt], convert_to_numpy=True).astype("float32")
    _, ids = index.search(q_vec, 4)
    ctx = "\n\n---\n\n".join([chunks[i] for i in ids[0] if 0 <= i < len(chunks)])

    client = Groq(api_key=st.session_state.api_key)
    return client.chat.completions.create(
        model=st.session_state.model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT.format(context=ctx)},
                  {"role": "user", "content": prompt}],
        temperature=0.15, max_tokens=1024
    ).choices[0].message.content


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h3 style="color:#1ea7ef;margin:0;">🎓 CorvitRAG</h3><p style="color:#64748b;font-size:0.8rem;">Knowledge Assistant</p>',
        unsafe_allow_html=True)
    st.session_state.api_key = st.text_input("API Key", type="password", value=st.session_state.api_key,
                                             placeholder="gsk_…").strip()
    st.session_state.model = st.selectbox("Model", GROQ_MODELS)

    col1, col2 = st.columns(2)
    connect_btn = col1.button("Connect", type="primary", use_container_width=True)
    rebuild_btn = col2.button("Rebuild", use_container_width=True)
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [];
        st.rerun()

    status_color = "#39ce7a" if st.session_state.connected else "#8491a2"
    status_text = f"Connected ({st.session_state.n_chunks} chunks)" if st.session_state.connected else "Not connected"
    st.markdown(f'<p style="color:{status_color};font-size:0.8rem;margin-top:10px;">● {status_text}</p>',
                unsafe_allow_html=True)

if connect_btn or rebuild_btn:
    if not st.session_state.api_key:
        st.sidebar.error("Missing API Key")
    else:
        try:
            Groq(api_key=st.session_state.api_key).chat.completions.create(model=st.session_state.model,
                                                                           messages=[{"role": "user", "content": "hi"}],
                                                                           max_tokens=1)
            index, chunks = build_or_load_index(force=rebuild_btn)
            st.session_state.update(
                {"index": index, "chunks": chunks, "n_chunks": len(chunks), "embedder": get_embedder(),
                 "connected": True})
            st.rerun()
        except Exception as e:
            st.session_state.connected = False;
            st.sidebar.error(str(e))

# ── Main Chat Interface ────────────────────────────────────────────────────
st.markdown('<h2 style="margin:0;">Corvit<span style="color:#1ea7ef;">RAG</span></h2>', unsafe_allow_html=True)

if st.session_state.connected and not st.session_state.messages:
    cols = st.columns(3)
    suggestions = ["AI courses offered?", "Rawalpindi campus?", "CEH & AWS prices?", "Certificates issued?", "Timings?",
                   "PSEB programs?"]
    for i, q in enumerate(suggestions):
        if cols[i % 3].button(q, key=f"s{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q});
            st.rerun()

for m in st.session_state.messages:
    st.markdown(f'<div class="{"msg-user" if m["role"] == "user" else "msg-bot"}">{m["content"]}</div>',
                unsafe_allow_html=True)

if prompt := st.chat_input("Ask about Corvit…"):
    if not st.session_state.connected:
        st.warning("Please connect first via sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

if st.session_state.connected and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Processing..."):
        try:
            ans = query_rag(st.session_state.messages[-1]["content"])
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.rerun()
        except Exception as e:
            st.error(str(e))