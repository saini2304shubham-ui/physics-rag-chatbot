#!/usr/bin/env python3
"""
scripts/download_corpus.py
────────────────────────────────────────────────────────────
Downloads free, open-licensed physics textbooks for the corpus.
All books are from OpenStax (CC-BY 4.0) or similar open sources.

Usage:
    python scripts/download_corpus.py
    python scripts/download_corpus.py --list      # just print the list
"""
import argparse
import sys
import urllib.request
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent / "corpus"

# ── Open-access physics textbooks (OpenStax CC-BY 4.0) ─────────────────
CORPUS = [
    {
        "name": "University Physics Vol 1 (Mechanics)",
        "filename": "university_physics_vol1.pdf",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume1-WEB_7Zesafu.pdf",
        "topics": ["mechanics", "kinematics", "dynamics", "thermodynamics"],
    },
    {
        "name": "University Physics Vol 2 (Thermodynamics, E&M, Optics)",
        "filename": "university_physics_vol2.pdf",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume2-WEB_Mg9C0Dm.pdf",
        "topics": ["thermodynamics", "electromagnetism", "optics"],
    },
    {
        "name": "University Physics Vol 3 (Modern Physics)",
        "filename": "university_physics_vol3.pdf",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume3-WEB.pdf",
        "topics": ["quantum mechanics", "nuclear physics", "relativity"],
    },
]


def download_with_progress(url: str, dest: Path):
    """Download a file showing progress."""
    downloaded = 0

    def progress(block_num, block_size, total_size):
        nonlocal downloaded
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb  = downloaded / 1e6
            print(f"\r  {pct:5.1f}%  {mb:.1f} MB", end="", flush=True)

    print(f"  Downloading → {dest.name}")
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\r  ✅ Done ({dest.stat().st_size / 1e6:.1f} MB)        ")


def main():
    parser = argparse.ArgumentParser(description="Download physics corpus PDFs")
    parser.add_argument("--list", action="store_true", help="List books without downloading")
    args = parser.parse_args()

    print("\n📚 Physics RAG — Corpus Downloader")
    print("   Source: OpenStax (CC-BY 4.0 — free to use)\n")

    for book in CORPUS:
        topics = ", ".join(book["topics"])
        print(f"  [{book['filename']}]  {book['name']}")
        print(f"    Topics: {topics}")

    if args.list:
        return

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving to: {CORPUS_DIR.resolve()}\n")

    for book in CORPUS:
        dest = CORPUS_DIR / book["filename"]
        if dest.exists():
            print(f"  ✓ Already exists: {book['filename']} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        try:
            download_with_progress(book["url"], dest)
        except Exception as e:
            print(f"\n  ❌ Failed to download {book['filename']}: {e}")
            print(f"     Please download manually from: {book['url']}")
            print(f"     and save to: {dest}")

    print("\n✅ Corpus ready! Now run:")
    print("   python -m src.ingestion.ingest --reset\n")


if __name__ == "__main__":
    main()
