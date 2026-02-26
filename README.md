# 🧠 Dynamic RAG Assistant

> **Your documents. Your machine. Your answers.**

A production-grade, fully local Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and ask natural language questions about it — with every byte of processing happening on your own hardware. No cloud. No API costs. No data leaving your device.

Built and optimised for CPU-only inference on consumer hardware (Intel i7-1355U). Response time reduced from an initial **112 seconds** to **20–35 seconds** through systematic architecture decisions and runtime tuning.

---

## 📸 Demo

<!-- After recording your demo, place it in an assets/ folder and update the path below -->
![LocalMind Demo](assets/demo.gif)

> *Upload a PDF → Ask questions → Get accurate, streaming answers in under 35 seconds, entirely on-device.*

---

## ✨ Features

- 🔒 **100% Local & Private** — No data ever leaves your machine. No API keys required.
- 🔍 **Hybrid Retrieval** — Combines dense vector search (ChromaDB) with sparse keyword search (BM25) for maximum recall.
- 🎯 **Cross-Encoder Reranking** — A precision second-pass reranker scores each result against your exact question before sending to the LLM.
- 📄 **Parent-Child Chunking** — Small 300-char child chunks for precise retrieval; full 1000-char parent sections sent to the LLM for coherent, complete answers.
- ⚡ **Streaming Responses** — Token-by-token rendering with a live cursor. First word appears in 5–10 seconds.
- 🧮 **Dynamic Token Budgeting** — Automatically allocates 150 / 250 / 450 tokens based on query type, preventing both answer cutoff and wasted compute.
- 📊 **Real-Time Ingestion Progress** — Per-step progress bars show exactly where time is being spent during document indexing.
- 🛠️ **Debug Inspector** — Built-in panel to inspect raw retrieval hits and reranked context windows side-by-side.

---

## 🏗️ Architecture

![Architecture Diagram](assets/Screenshot%202026-02-26%20182130.png)


---

## ⚡ Performance

> All benchmarks on Intel i7-1355U · 16 GB RAM · No GPU

| Metric | Result |
|--------|--------|
| PDF Ingestion (60 pages) | ~10–15 seconds |
| Hybrid Retrieval | < 1.0 second |
| Time to First Token | 5–10 seconds |
| Full Response — simple query | ~20 seconds |
| Full Response — list / enumeration query | ~30–35 seconds |
| Response time before optimisation | 112 seconds |
| **Total speedup achieved** | **~4–5×** |

---

## 🔧 Tech Stack

| Category | Technology | Why This — Not the Alternative |
|----------|-----------|-------------------------------|
| **LLM** | Qwen 2.5 (3B) via Ollama | Best tokens/sec on CPU at sub-4B size. Tested `llama3:8b` → 112 s ❌ and `phi3:mini` → 65 s ❌ |
| **Embeddings** | nomic-embed-text (768-dim) | Fast CPU inference. `snowflake-arctic-embed2` (1024-dim) was 3× slower on batch ingestion ❌ |
| **Reranker** | ms-marco-MiniLM-L-4-v2 | 4-layer distilled model, ~0.5 s. `bge-reranker-base` was 4.4 s at 12 layers ❌ |
| **Vector DB** | ChromaDB | Fully local, persistent, LangChain-native, no server required |
| **Sparse Search** | BM25 via rank_bm25 | Catches exact terms (drug names, codes, acronyms) that semantic vectors miss |
| **PDF Parser** | PyMuPDF4LLM | Native parsing, ~10 s/60 pages. `marker-pdf` loaded 4 ML models → 45 minutes ❌ |
| **Orchestration** | LangChain + Ollama | Local model serving + pipeline wiring with streaming support |
| **UI** | Streamlit | Built-in streaming, reactive components, fast iteration |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- 8 GB+ RAM (16 GB recommended)
- Windows / macOS / Linux

### 1 · Clone the repository

```bash
git clone https://github.com/yourusername/localmind.git
cd localmind
```

### 2 · Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Pull required Ollama models

```bash
# Make sure Ollama is running first
ollama serve

# In a new terminal
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 5 · Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📖 Usage

1. **Upload a PDF** using the sidebar file uploader
2. **Wait for ingestion** (~10–15 s for a typical document) — per-step progress is shown
3. **Ask questions** in the chat input at the bottom
4. **Watch the answer stream** word by word — first token appears in 5–10 seconds
5. **Clear and reset** using the 🗑️ button in the sidebar to start a new session

### Example queries that work well

```
What are the main topics covered in this document?
List all ECOG performance statuses and describe each one.
What are the side effects of Cisplatin?
Describe all Framework Implementation Tiers in detail.
What does the document say about cardiac monitoring requirements?
Summarise the key recommendations in section 3.
```

---

## 📁 Project Structure

```
localmind/
│
├── app.py                  # Streamlit UI, session state, chat interface, streaming
├── config.py               # All configuration constants (models, chunk sizes, paths)
├── database.py             # ChromaDB + BM25 indexing, hybrid retrieval, reranking
├── llm_engine.py           # LLM singleton, prompt template, dynamic token budgeting
├── utils.py                # PDF parsing, chunking pipeline, parallel processing
├── requirements.txt        # Python dependencies
│
├── assets/                 # Screenshots and demo GIF for README
├── chroma_db_storage/      # Persisted vector database  ← auto-created, gitignored
└── temp_uploads/           # Temporary PDF storage      ← auto-created, gitignored
```

---

## ⚙️ Configuration

All tuneable parameters are in `config.py`:

```python
class Config:
    # ── Models ────────────────────────────────────────
    LLM_MODEL        = "qwen2.5:3b"
    EMBEDDING_MODEL  = "nomic-embed-text"

    # ── Chunking ──────────────────────────────────────
    PARENT_CHUNK_SIZE    = 1000   # Characters per parent section
    PARENT_CHUNK_OVERLAP = 100
    CHILD_CHUNK_SIZE     = 300    # Characters per child chunk (for retrieval)
    CHILD_CHUNK_OVERLAP  = 30

    # ── Retrieval ─────────────────────────────────────
    RETRIEVAL_K       = 6         # Candidates pulled from hybrid search
    RERANK_TOP_N      = 2         # Final sections passed to LLM
    MAX_CONTEXT_CHARS = 2000      # Hard cap on total context sent to LLM

    # ── LLM Runtime ───────────────────────────────────
    LLM_NUM_CTX     = 1536        # Context window (right-sized for 2000-char budget)
    LLM_NUM_THREAD  = 12          # Set to your CPU's thread count
    LLM_TEMPERATURE = 0.1
```

**Tuning tips**

| Goal | Change |
|------|--------|
| Faster responses | Lower `MAX_CONTEXT_CHARS` to 1400, set `RERANK_TOP_N = 2` |
| More detailed answers | Raise `MAX_CONTEXT_CHARS` to 3000, set `RERANK_TOP_N = 3` |
| Better CPU utilisation | Set `LLM_NUM_THREAD` to your exact thread count |
| Reduce answer cutoff on lists | Increase `num_predict` in `llm_engine.py` for list queries |

---

## 🧠 How It Works

### Ingestion Phase *(runs once per document)*

| Step | What happens |
|------|-------------|
| **1 · Parse** | PyMuPDF4LLM reads native PDF structure (font sizes, coordinates) to produce clean Markdown — no ML models, no OCR |
| **2 · Parent split** | `MarkdownHeaderTextSplitter` cuts on `#` / `##` / `###` headers → 30–80 coherent parent sections |
| **3 · Child chunks** | Each parent split into 300-char child chunks across 4 parallel threads. Each child stores a reference back to its full parent |
| **4 · Batch embed** | Child texts sent to `nomic-embed-text` in batches of 64 — collapses 200 HTTP round-trips into just 4 calls |
| **5 · Index** | Pre-computed embeddings inserted into ChromaDB in one batch call; BM25 index built simultaneously |
| **6 · Warm-up** | Reranker and LLM loaded into RAM — eliminates cold-start penalty on first question |

### Retrieval + Generation Phase *(runs per question)*

| Step | What happens |
|------|-------------|
| **1 · Classify** | Query scanned for list vs fact keywords → `num_predict` set to 450 / 250 / 150 accordingly |
| **2 · Dense search** | Question embedded → cosine similarity against all chunks → top 6 semantically similar chunks |
| **3 · Sparse search** | Question tokenised → BM25 scores computed → top 6 keyword matches |
| **4 · Fuse** | Results merged and deduplicated by content fingerprint → up to 12 candidates |
| **5 · Rerank** | Cross-encoder scores each `(question, chunk)` pair jointly → sorted by precise relevance score |
| **6 · Promote** | Top 2–3 chunks retrieve their full parent sections; 4 most relevant sentences extracted per parent |
| **7 · Generate** | Context + question → prompt template → Ollama streams tokens → Streamlit renders each token live with cursor |

---

## 🚧 Known Limitations

| Limitation | Detail |
|-----------|--------|
| **Response speed** | 20–35 s is a CPU hardware ceiling, not a code problem. GPU reduces this to 1–3 s |
| **Single document** | One PDF per session. No cross-document querying currently supported |
| **No OCR** | Scanned / image-only PDFs return empty extraction. Only native-text PDFs are supported |
| **No persistence** | Clearing the session wipes everything. Re-upload required each new session |
| **3B model reasoning** | Complex multi-step reasoning may produce partial answers. 7B+ models handle this better |
| **Windows file locks** | ChromaDB holds file handles on Windows. Session reset includes a retry loop to handle this gracefully |

---

## 🔮 Roadmap

- [ ] GPU acceleration (CUDA / Vulkan) → sub-3 s responses, unlock 7B+ models
- [ ] Multi-document support — persistent cross-file knowledge base with document library UI
- [ ] OCR integration — Docling / Tesseract fallback for scanned PDFs
- [ ] Session persistence — SQLite-backed sessions with saved chat history
- [ ] Agentic multi-hop retrieval — LangGraph-based planning for complex cross-section questions
- [ ] RAGAS evaluation framework — automated faithfulness and relevance scoring
- [ ] Knowledge graph layer — entity relationship mapping across documents

---

## 📦 Dependencies

```
streamlit>=1.35.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-ollama>=0.1.0
langchain-core>=0.2.0
langchain-text-splitters>=0.2.0
chromadb>=0.5.0
pymupdf4llm>=0.0.17
pymupdf>=1.24.0
sentence-transformers>=3.0.0
rank_bm25>=0.2.2
python-dotenv>=1.0.0
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create your feature branch — `git checkout -b feature/your-feature`
3. Commit your changes — `git commit -m 'Add your feature'`
4. Push to the branch — `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 👤 Author

**Chirag Dugar** · 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](www.linkedin.com/in/chirag-dugar-341077252)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/chiragdugar04)

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.ai) — Local model serving
- [LangChain](https://langchain.com) — Pipeline orchestration
- [ChromaDB](https://trychroma.com) — Vector storage
- [Qwen 2.5](https://huggingface.co/Qwen) — Language model by Alibaba
- [BAAI](https://huggingface.co/BAAI) — BGE reranker models
- [Nomic AI](https://nomic.ai) — nomic-embed-text embedding model

---

<p align="center">
  <sub>Built with ❤️ on a CPU. No GPU required.</sub>
</p>
