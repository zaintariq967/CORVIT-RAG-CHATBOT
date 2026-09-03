# CorvitRAG – Professional RAG Chatbot for Corvit Systems

A production-ready **Retrieval-Augmented Generation** chatbot that answers questions **only** from the official Corvit Systems knowledge-base PDF.

**Stack:** Streamlit · LangChain · FAISS · HuggingFace Embeddings · Groq (Llama-3)

---

## Project Structure

```
Corvit-RAG-Chatbot/
├── asset/
│   └── data/
│       └── corvit.pdf          # Knowledge-base PDF (already placed)
├── main.py                     # Single entry-point application
├── requirements.txt
└── README.md
```

On first run a `asset/vectorstore/` folder is created automatically and the FAISS index is persisted.

---

## How to Run (PyCharm / Terminal)

### 1. Create & activate virtual environment (recommended)

```bash
cd Corvit-RAG-Chatbot
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First install may take a few minutes because of `torch` + `sentence-transformers`.

### 3. Get a free Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up / Log in
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

### 4. Launch the app

```bash
streamlit run main.py
```

The browser will open automatically (usually `http://localhost:8501`).

### 5. Enter API Key at runtime

- Paste the Groq API key in the **sidebar** (password field).
- The key is kept only in the current Streamlit session – it is **never** written to disk.
- Choose the model (default: `llama-3.3-70b-versatile`).

---

## Architecture & Workflow

```
User Question
      │
      ▼
┌─────────────────┐
│  Streamlit UI   │  ← professional dark theme, chat bubbles, suggested questions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │
│  1. Retrieve    │  FAISS similarity search (top-4 chunks)
│  2. Augment     │  Inject retrieved context into system prompt
│  3. Generate    │  Groq LLM (grounded, temperature 0.15)
└────────┬────────┘
         │
         ▼
   Grounded Answer
```

### Detailed Steps

| Step | Component | Description |
|------|-----------|-------------|
| 1 | PDF Loader | `PyPDFLoader` extracts text from `asset/data/corvit.pdf` |
| 2 | Chunking | `RecursiveCharacterTextSplitter` (800 tokens, 150 overlap) |
| 3 | Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| 4 | Vector Store | FAISS – saved to disk after first build |
| 5 | Retrieval | Top-4 most similar chunks |
| 6 | LLM | Groq Chat model with strict grounded system prompt |
| 7 | UI | Streamlit – dark professional theme, sidebar config, chat history |

---

## Key Design Decisions

- **API key at runtime only** – no `.env` file required; enter in sidebar.
- **No hallucination on prices/schedules** – system prompt forces the model to say “not found” when data is missing.
- **Persistent index** – first run builds the vector store; subsequent runs load instantly.
- **Rebuild button** – force re-index if you update the PDF.
- **Suggested questions** – zero-click start for new users.
- **Single file** – everything lives in `main.py` for easy PyCharm import.

---

## Sample Questions the Bot Can Answer

- What AI courses does Corvit offer?
- Where is the Rawalpindi campus and its phone number?
- What is the catalog price of CEH / AWS SAA-C03 / AI Deep Learning?
- Does Corvit issue certificates?
- What are the general timings?
- Which PSEB trainings are listed?
- Payment methods for onsite students?

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `PDF not found` | Ensure `corvit.pdf` is inside `asset/data/` |
| `Invalid API Key` | Double-check the key from console.groq.com |
| Slow first run | Embedding model download (~90 MB) happens once |
| `torch` install error | Use `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

---

## License & Disclaimer

This chatbot is an educational / internal knowledge assistant built on publicly available Corvit information.  
Prices, phone numbers and schedules can change — always verify on the official Corvit website.
