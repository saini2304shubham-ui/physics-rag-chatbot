"""
ui/app.py — Streamlit Chat Interface for Physics RAG Chatbot
────────────────────────────────────────────────────────────
Run with:
    streamlit run src/ui/app.py
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ─────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PhysicsBot — RAG Chatbot",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.retrieval.retriever import PhysicsRAGChain
from src.config import LLM_MODEL, COLLECTION_NAME, TOP_K_RESULTS

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:   #0d1117;
    --bg-secondary: #161b22;
    --bg-card:      #21262d;
    --accent:       #58a6ff;
    --accent2:      #39d353;
    --warn:         #f85149;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --border:       #30363d;
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Chat Messages ── */
.user-bubble {
    background: linear-gradient(135deg, #1f4068, #1a3350);
    border: 1px solid #2d5a8e;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 60px;
    color: #cdd9e5;
    font-size: 0.95rem;
    line-height: 1.6;
}
.bot-bubble {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px;
    margin: 8px 60px 8px 0;
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.7;
}
.bot-bubble code {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85em;
    color: #79c0ff;
}

/* ── Source Cards ── */
.source-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
    color: var(--text-muted);
}
.source-card .source-title {
    font-weight: 600;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin-bottom: 4px;
}
.source-snippet {
    font-style: italic;
    line-height: 1.5;
    color: #8d96a0;
}

/* ── Confidence Badge ── */
.conf-badge-high   { background:#1a4731; border:1px solid #39d353; color:#39d353; }
.conf-badge-medium { background:#332d00; border:1px solid #d29922; color:#d29922; }
.conf-badge-low    { background:#3b1219; border:1px solid #f85149; color:#f85149; }
.conf-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    margin-left: 8px;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 20px 0 10px 0;
}
.app-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: var(--accent);
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.app-header p {
    color: var(--text-muted);
    font-size: 0.9rem;
}

/* ── Out-of-scope banner ── */
.oos-banner {
    background: #3b1219;
    border: 1px solid var(--warn);
    border-radius: 8px;
    padding: 12px 16px;
    color: #ffa198;
    font-size: 0.9rem;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Input ── */
.stTextInput > div > div > input,
.stChatInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
}

/* ── Expander ── */
details {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
if "in_scope_count" not in st.session_state:
    st.session_state.in_scope_count = 0
if "avg_confidence" not in st.session_state:
    st.session_state.avg_confidence = []


# ─────────────────────────────────────────
# LOAD RAG CHAIN
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading PhysicsBot…")
def load_chain():
    try:
        return PhysicsRAGChain()
    except FileNotFoundError as e:
        return str(e)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚛️ PhysicsBot")
    st.markdown("---")

    # Stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Questions", st.session_state.total_questions)
    col2.metric(
        "In-scope",
        f"{st.session_state.in_scope_count}/{st.session_state.total_questions}"
        if st.session_state.total_questions else "—"
    )
    if st.session_state.avg_confidence:
        avg = sum(st.session_state.avg_confidence) / len(st.session_state.avg_confidence)
        st.metric("Avg Confidence", f"{avg:.0%}")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.markdown(f"**Model:** `{LLM_MODEL}`")
    st.markdown(f"**Collection:** `{COLLECTION_NAME}`")
    st.markdown(f"**Top-K:** `{TOP_K_RESULTS}`")

    st.markdown("---")
    st.markdown("### 🔬 Example Questions")
    example_questions = [
        "What is Newton's second law?",
        "Explain the photoelectric effect",
        "Derive the kinematic equations",
        "What is Schrödinger's equation?",
        "How does a capacitor store energy?",
        "What is the second law of thermodynamics?",
        "Explain wave-particle duality",
        "What is Snell's law of refraction?",
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_questions = 0
        st.session_state.in_scope_count = 0
        st.session_state.avg_confidence = []
        st.rerun()


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>⚛️ PhysicsBot</h1>
  <p>RAG-powered undergraduate physics tutor · Every answer is grounded in source texts</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD CHAIN + ERROR STATE
# ─────────────────────────────────────────
chain_or_error = load_chain()
if isinstance(chain_or_error, str):
    st.error(f"""
**Vector store not found.**

Run the ingestion pipeline first:
```bash
python -m src.ingestion.ingest
```

Then restart the app.

Error: `{chain_or_error}`
""")
    st.stop()
else:
    rag_chain = chain_or_error


# ─────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────
def render_confidence_badge(conf: float) -> str:
    if conf >= 0.7:
        cls, label = "conf-badge-high",   f"✓ {conf:.0%} confidence"
    elif conf >= 0.4:
        cls, label = "conf-badge-medium", f"~ {conf:.0%} confidence"
    else:
        cls, label = "conf-badge-low",    f"! {conf:.0%} confidence"
    return f'<span class="conf-badge {cls}">{label}</span>'


for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑‍🎓 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        # Bot message
        badge = render_confidence_badge(msg.get("confidence", 0))
        in_scope = msg.get("in_scope", True)

        if not in_scope:
            st.markdown(f'<div class="oos-banner">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="bot-bubble">🤖 {badge}<br><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )

        # Source citations
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📚 {len(sources)} source passage(s) used", expanded=False):
                for src in sources:
                    st.markdown(f"""
<div class="source-card">
  <div class="source-title">📄 {src['source']} · Page {src['page']}</div>
  <div class="source-snippet">"{src['snippet']}"</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HANDLE PENDING EXAMPLE QUESTION
# ─────────────────────────────────────────
pending = st.session_state.pop("pending_question", None)


# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
user_input = st.chat_input("Ask a physics question…") or pending

if user_input:
    # Display user message
    st.markdown(f'<div class="user-bubble">🧑‍🎓 {user_input}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.total_questions += 1

    # Run RAG chain
    with st.spinner("Searching corpus and generating answer…"):
        result = rag_chain.answer(user_input)

    # Track stats
    if result["in_scope"]:
        st.session_state.in_scope_count += 1
        st.session_state.avg_confidence.append(result["confidence"])

    # Display answer
    badge = render_confidence_badge(result["confidence"])
    if not result["in_scope"]:
        st.markdown(f'<div class="oos-banner">{result["answer"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="bot-bubble">🤖 {badge}<br><br>{result["answer"]}</div>',
            unsafe_allow_html=True
        )

    # Source citations
    if result["sources"]:
        with st.expander(f"📚 {len(result['sources'])} source passage(s) used", expanded=True):
            for src in result["sources"]:
                st.markdown(f"""
<div class="source-card">
  <div class="source-title">📄 {src['source']} · Page {src['page']}</div>
  <div class="source-snippet">"{src['snippet']}"</div>
</div>""", unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    result["answer"],
        "sources":    result["sources"],
        "confidence": result["confidence"],
        "in_scope":   result["in_scope"],
    })

    st.rerun()
