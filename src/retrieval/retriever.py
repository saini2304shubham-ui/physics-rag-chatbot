import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    CHROMA_DIR, COLLECTION_NAME, LLM_MODEL,
    GROQ_API_KEY, TOP_K_RESULTS, RELEVANCE_THRESHOLD, PHYSICS_KEYWORDS,
)

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are PhysicsBot, an expert undergraduate physics tutor.
Answer physics questions using ONLY the provided source passages.

RULES:
1. Base your answer exclusively on the [SOURCE] passages below.
2. After EVERY factual claim, add an inline citation: [Source: <filename>, p.<page>]
3. If passages don't contain enough information, say: "I don't have sufficient information in my corpus to fully answer this."
4. Never fabricate formulas, constants, or derivations not present in the passages.
5. Use clear, pedagogical language suitable for an undergraduate student.

SOURCE PASSAGES:
{context}
"""

RAG_HUMAN_PROMPT = """Question: {question}

Please answer using only the source passages above. Include inline citations for every fact."""

OUT_OF_SCOPE_RESPONSE = (
    "⚠️ **Out of Scope**: This question does not appear to be about undergraduate physics. "
    "I am specialized in physics topics only (mechanics, electromagnetism, thermodynamics, "
    "quantum mechanics, optics, nuclear physics, etc.).\n\n"
    "Please ask a physics-related question and I will do my best to help!"
)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore() -> Chroma:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_DIR}. "
            "Run 'python -m src.ingestion.ingest' first."
        )
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def is_physics_question(question: str, retrieved_docs: List[Document]) -> Tuple[bool, float]:
    q_lower = question.lower()
    keyword_hits = sum(1 for kw in PHYSICS_KEYWORDS if kw in q_lower)
    if keyword_hits >= 2:
        return True, min(0.95, 0.5 + keyword_hits * 0.05)
    if not retrieved_docs:
        return False, 0.0
    if keyword_hits == 1:
        return True, 0.65
    return False, 0.1


def format_context(docs: List[Document]) -> Tuple[str, List[Dict]]:
    context_parts = []
    sources = []
    for i, doc in enumerate(docs, start=1):
        meta    = doc.metadata
        src     = meta.get("source", "Unknown")
        page    = meta.get("page_num", "?")
        snippet = doc.page_content.strip()
        context_parts.append(f"[PASSAGE {i}] Source: {src}, Page {page}\n{snippet}")
        sources.append({
            "index": i,
            "source": src,
            "page": page,
            "snippet": snippet[:300] + ("..." if len(snippet) > 300 else ""),
            "full_text": snippet,
        })
    return "\n\n---\n\n".join(context_parts), sources


def compute_confidence(docs: List[Document], scores: Optional[List[float]], is_in_scope: bool) -> float:
    if not is_in_scope or not docs:
        return 0.0
    n = len(docs)
    base = min(1.0, n / TOP_K_RESULTS)
    score_factor = (sum(scores) / len(scores)) if scores else 0.6
    return round(min(0.4 * base + 0.6 * score_factor, 0.99), 2)


class PhysicsRAGChain:
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore or load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS},
        )
        self.llm = ChatGroq(
            model=LLM_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.1,
            max_tokens=1500,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human",  RAG_HUMAN_PROMPT),
        ])

    def retrieve_with_scores(self, question: str) -> Tuple[List[Document], Optional[List[float]]]:
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                question, k=TOP_K_RESULTS
            )
            if not results:
                return [], []
            docs, scores = zip(*results)
            filtered = [(d, s) for d, s in zip(docs, scores) if s >= RELEVANCE_THRESHOLD]
            if not filtered:
                return [], []
            docs, scores = zip(*filtered)
            return list(docs), list(scores)
        except Exception as e:
            logger.warning(f"Scored retrieval failed: {e}")
            docs = self.retriever.invoke(question)
            return docs, None

    def answer(self, question: str) -> Dict:
        docs, scores = self.retrieve_with_scores(question)
        in_scope, _  = is_physics_question(question, docs)

        if not in_scope:
            return {
                "answer": OUT_OF_SCOPE_RESPONSE,
                "sources": [], "confidence": 0.0,
                "in_scope": False, "question": question,
            }

        if not docs:
            return {
                "answer": "I couldn't find relevant passages in my physics corpus for this question.",
                "sources": [], "confidence": 0.1,
                "in_scope": True, "question": question,
            }

        context, sources = format_context(docs)
        chain  = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        confidence = compute_confidence(docs, scores, in_scope)

        return {
            "answer": answer, "sources": sources,
            "confidence": confidence, "in_scope": True, "question": question,
        }