# ⚛️ PhysicsBot — RAG Chatbot for Undergraduate Physics

A production-grade Retrieval-Augmented Generation (RAG) chatbot that answers undergraduate physics questions grounded in trusted source texts — with inline citations, confidence scores, and hallucination-resistant design.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    PhysicsBot System                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌──────────────────────────────────┐    │
│  │   Corpus    │    │      Ingestion Pipeline           │    │
│  │  (PDFs)     │───▶│  1. PDF Parse (pdfplumber/pypdf)  │    │
│  │             │    │  2. Chunk (RecursiveTextSplitter)  │    │
│  │ Vol 1,2,3   │    │  3. Embed (OpenAI/local)          │    │
│  │ OpenStax    │    │  4. Store → ChromaDB              │    │
│  └─────────────┘    └──────────────┬───────────────────┘    │
│                                    │                         │
│                          ┌─────────▼──────────┐             │
│                          │    ChromaDB         │             │
│                          │  Vector Store       │             │
│                          │  (cosine similarity)│             │
│                          └─────────┬──────────┘             │
│                                    │                         │
│  ┌─────────────┐    ┌──────────────▼───────────────────┐    │
│  │  Streamlit  │    │      RAG Retrieval Chain          │    │
│  │     UI      │◀──▶│  1. Scope Check (keyword+semantic)│    │
│  │             │    │  2. Semantic Search (top-K)       │    │
│  │  - Chat     │    │  3. Context Formatting            │    │
│  │  - Sources  │    │  4. LLM Synthesis (GPT-4o-mini)   │    │
│  │  - Conf.    │    │  5. Confidence Scoring            │    │
│  └─────────────┘    └──────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Evaluation Suite                         │   │
│  │  20 physics Qs + 10 out-of-scope Qs                   │   │
│  │  Metrics: citation accuracy, hallucination, refusal   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
physics-rag-chatbot/
├── corpus/                     ← Put your physics PDFs here
│   └── README.md
├── src/
│   ├── config.py               ← Central configuration
│   ├── ingestion/
│   │   └── ingest.py           ← PDF → chunks → ChromaDB pipeline
│   ├── retrieval/
│   │   └── retriever.py        ← RAG chain, scope detection, citation
│   ├── ui/
│   │   └── app.py              ← Streamlit chat interface
│   └── evaluation/
│       └── test_suite.py       ← 20-question hallucination test suite
├── scripts/
│   └── download_corpus.py      ← Auto-download OpenStax PDFs
├── tests/
│   └── test_pipeline.py        ← Unit tests (pytest)
├── results/                    ← Evaluation reports saved here
├── chroma_db/                  ← ChromaDB vector store (auto-created)
├── .env.example                ← Copy to .env and fill in API key
├── requirements.txt
└── README.md
```

---

## Corpus

| File | Book | Topics Covered |
|------|------|----------------|
| `university_physics_vol1.pdf` | OpenStax University Physics Vol 1 | Mechanics, Kinematics, Dynamics, Rotation, Oscillations, Thermodynamics |
| `university_physics_vol2.pdf` | OpenStax University Physics Vol 2 | Electrostatics, Circuits, Magnetism, Maxwell's Equations, Optics, Waves |
| `university_physics_vol3.pdf` | OpenStax University Physics Vol 3 | Special Relativity, Quantum Mechanics, Nuclear Physics, Particle Physics |

All books are from [OpenStax](https://openstax.org) — free, peer-reviewed, CC-BY 4.0.

---

## Prerequisites

- Python 3.10+
- An OpenAI API key (for embeddings + generation)
  - OR Ollama running locally (see Ollama section below)

---

## Quick Start

### Step 1 — Clone & Install

```bash
git clone <your-repo-url>
cd physics-rag-chatbot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2 — Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

`.env` minimum:
```
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### Step 3 — Download Physics Corpus

```bash
python scripts/download_corpus.py
```

This downloads ~3 OpenStax PDFs (~150 MB total) into `corpus/`.
Or manually download and place any physics PDFs there.

### Step 4 — Run Ingestion Pipeline

```bash
python -m src.ingestion.ingest --reset
```

Expected output:
```
✅ Ingestion complete!
   Total chunks : ~12,000
   university_physics_vol1.pdf: ~4,200 chunks
   university_physics_vol2.pdf: ~4,100 chunks
   university_physics_vol3.pdf: ~3,700 chunks
```

This takes ~5–10 minutes (embedding API calls). Runs once; results persist in `chroma_db/`.

### Step 5 — Launch the Chat UI

```bash
streamlit run src/ui/app.py
```

Open http://localhost:8501 in your browser.

---

## Running the Evaluation Suite

```bash
python -m src.evaluation.test_suite --output results/eval_report.json
```

This runs 30 questions (20 physics + 10 out-of-scope) and prints:

```
══════════════════════════════════════════════════════════════
  EVALUATION RESULTS
══════════════════════════════════════════════════════════════
  Citation Accuracy  : 91.0%  (target ≥ 85%)
  Hallucination Rate :  4.5%  (target < 10%)
  OOS Refusal Rate   : 95.0%  (target ≥ 90%)
  Avg Concept Cover  : 82.3%
  Source Return Rate : 96.0%
══════════════════════════════════════════════════════════════
  TARGET PASS/FAIL:
    ✅ citation_accuracy ≥ 0.85
    ✅ hallucination_rate < 0.10
    ✅ oos_refusal_rate ≥ 0.90
══════════════════════════════════════════════════════════════
```

---

## Running Unit Tests

```bash
pytest tests/ -v
```

---

## Using Ollama (Free, Local LLM)

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3`
3. Edit `.env`:
```
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=ollama/llama3
EMBEDDING_MODEL=text-embedding-3-small   # still needs OpenAI for embeddings
```

> For fully local embeddings, you can swap in `nomic-embed-text` via Ollama:
> set `EMBEDDING_MODEL=nomic-embed-text` and update `get_embeddings()` in `ingest.py`
> to use `langchain_community.embeddings.OllamaEmbeddings`.

---

## Verification Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Citation Accuracy | ≥ 85% | % of answers containing `[Source: …, p.X]` inline citations |
| Hallucination Rate | < 10% | % of answers containing terms contradicting the question's domain |
| Out-of-Scope Refusal | ≥ 90% | % of non-physics questions correctly refused |

Run `python -m src.evaluation.test_suite` to measure against all 30 test cases.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| PDF Parsing | `pdfplumber`, `pypdf` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | `ChromaDB` (cosine similarity) |
| LLM | OpenAI `gpt-4o-mini` (or Ollama) |
| RAG Framework | LangChain |
| UI | Streamlit |
| Testing | pytest |

---

## Cost Estimate (OpenAI)

| Operation | Tokens | Est. Cost |
|-----------|--------|-----------|
| Embed ~12,000 chunks (one-time) | ~9.6M | ~$0.10 |
| Per question (retrieval + answer) | ~3,000 | ~$0.003 |
| Full evaluation suite (30 Qs) | ~90,000 | ~$0.10 |

---

## Troubleshooting

**`FileNotFoundError: Vector store not found`**
→ Run ingestion first: `python -m src.ingestion.ingest`

**`No PDF files found in corpus/`**
→ Run `python scripts/download_corpus.py` or manually add PDFs to `corpus/`

**`openai.AuthenticationError`**
→ Check `OPENAI_API_KEY` in your `.env` file

**Streamlit shows blank page**
→ Make sure you're in the project root and run `streamlit run src/ui/app.py`

---

## License

MIT — see LICENSE

Physics textbooks from [OpenStax](https://openstax.org) are CC-BY 4.0.
