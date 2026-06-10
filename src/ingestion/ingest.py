import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

import pdfplumber
from pypdf import PdfReader
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    CORPUS_DIR, CHROMA_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({
                        "page_num": i,
                        "text": text,
                        "source": pdf_path.name,
                        "source_path": str(pdf_path),
                    })
        if pages:
            return pages
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}. Trying pypdf...")

    try:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "page_num": i,
                    "text": text,
                    "source": pdf_path.name,
                    "source_path": str(pdf_path),
                })
    except Exception as e:
        logger.error(f"Both parsers failed: {e}")
    return pages


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = []
    for page in pages:
        raw_chunks = splitter.split_text(page["text"])
        for j, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < 50:
                continue
            chunk_id = hashlib.md5(
                f"{page['source']}_{page['page_num']}_{j}".encode()
            ).hexdigest()[:12]
            chunks.append({
                "id": chunk_id,
                "text": chunk_text.strip(),
                "metadata": {
                    "source": page["source"],
                    "page_num": page["page_num"],
                    "chunk_index": j,
                    "source_path": page["source_path"],
                },
            })
    return chunks


def get_embeddings():
    logger.info("Loading local embedding model (first time downloads ~90MB)...")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: List[Dict], reset: bool = False) -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if reset and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        logger.info("Wiped existing vector store.")
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    texts     = [c["text"]     for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids       = [c["id"]       for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks into ChromaDB...")
    batch_size = 500
    vectorstore = None
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[start:start + batch_size]
        batch_meta  = metadatas[start:start + batch_size]
        batch_ids   = ids[start:start + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_texts(
                texts=batch_texts,
                embedding=embeddings,
                metadatas=batch_meta,
                ids=batch_ids,
                persist_directory=str(CHROMA_DIR),
                collection_name=COLLECTION_NAME,
            )
        else:
            vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_meta,
                ids=batch_ids,
            )
    logger.info(f"Vector store saved to {CHROMA_DIR}")
    return vectorstore


def save_manifest(corpus_dir: Path, chunks: List[Dict]):
    from collections import Counter
    counts = Counter(c["metadata"]["source"] for c in chunks)
    manifest = {
        "total_chunks": len(chunks),
        "sources": [{"file": src, "chunks": cnt} for src, cnt in sorted(counts.items())],
    }
    with open(corpus_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run_ingestion(corpus_dir: Path = CORPUS_DIR, reset: bool = False):
    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {corpus_dir}.")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s)")
    all_chunks = []
    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        logger.info(f"Parsing: {pdf_path.name}")
        pages  = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages)
        logger.info(f"  → {len(pages)} pages, {len(chunks)} chunks")
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")
    build_vector_store(all_chunks, reset=reset)
    manifest = save_manifest(corpus_dir, all_chunks)

    print("\n✅ Ingestion complete!")
    print(f"   Total chunks : {manifest['total_chunks']}")
    for s in manifest["sources"]:
        print(f"   {s['file']}: {s['chunks']} chunks")
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", default=str(CORPUS_DIR))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run_ingestion(Path(args.corpus_dir), reset=args.reset)