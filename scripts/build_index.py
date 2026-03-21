#!/usr/bin/env python
"""
build_index.py — One-off script to scrape GitLab Handbook + Direction
and build the FAISS vector index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --force   # rebuild even if index exists

The index is saved to data/index/ (or INDEX_DIR env var).
Run this once before starting the server, or whenever you want fresh data.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Make sure we can import from backend/
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from rag.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Rebuild index even if it already exists.")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.build_index(force=args.force)
    print(f"\n✅ Index built — {pipeline.num_chunks} chunks stored.")

if __name__ == "__main__":
    main()
