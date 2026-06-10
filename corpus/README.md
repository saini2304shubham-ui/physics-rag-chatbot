# Physics Corpus

Place your physics textbook PDF files in this directory before running the ingestion pipeline.

## Recommended Free Textbooks (OpenStax — CC-BY 4.0)

Run the automatic downloader:
```bash
python scripts/download_corpus.py
```

This downloads three OpenStax volumes covering the full undergraduate curriculum:

| File | Topics |
|------|--------|
| `university_physics_vol1.pdf` | Mechanics, Kinematics, Thermodynamics |
| `university_physics_vol2.pdf` | Electromagnetism, Optics, Waves |
| `university_physics_vol3.pdf` | Quantum Mechanics, Nuclear, Relativity |

## Manual Download Links

If the script fails, download directly:
- Vol 1: https://openstax.org/details/books/university-physics-volume-1
- Vol 2: https://openstax.org/details/books/university-physics-volume-2
- Vol 3: https://openstax.org/details/books/university-physics-volume-3

## Adding Your Own PDFs

You can add any physics textbook PDFs to this folder. The ingestion pipeline
will parse all `*.pdf` files automatically. Name them descriptively so citations
are readable in the UI.

## After Adding Files

Run ingestion:
```bash
python -m src.ingestion.ingest --reset
```

Check the generated `manifest.json` to verify all PDFs were processed.
