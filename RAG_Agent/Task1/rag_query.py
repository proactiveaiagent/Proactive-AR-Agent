#!/usr/bin/env python3
"""Query the RAG knowledge graph"""

import sys

sys.path.append('/workspace/qwen/agents/RAG_agents')

from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
import os

# Set environment
os.chdir('/workspace/qwen/agents/RAG_agents')


# Custom embedding function
def local_embed_func(texts):
    model = SentenceTransformer('/workspace/qwen/agents/RAG_agents/models/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings


# Initialize RAG
rag = LightRAG(
    working_dir="./rag_storage",
    embedding_func=local_embed_func
)

print("=" * 70)
print("RAG KNOWLEDGE GRAPH QUERY INTERFACE")
print("=" * 70)
print("Type 'exit' to quit\n")

# Query modes
modes = {
    "1": "naive",  # Simple keyword search
    "2": "local",  # Entity-focused search
    "3": "global",  # High-level summary
    "4": "hybrid"  # Combined approach
}

while True:
    print("\nQuery Modes:")
    print("  1. Naive   - Simple keyword search")
    print("  2. Local   - Entity-focused (people, objects, places)")
    print("  3. Global  - High-level summary")
    print("  4. Hybrid  - Best of both worlds")

    mode_choice = input("\nSelect mode (1-4, default=4): ").strip() or "4"
    mode = modes.get(mode_choice, "hybrid")

    query = input(f"\nEnter query ({mode} mode): ").strip()

    if query.lower() in ['exit', 'quit', 'q']:
        break

    if not query:
        continue

    print(f"\n{'=' * 70}")
    print(f"QUERY: {query}")
    print(f"MODE: {mode}")
    print(f"{'=' * 70}\n")

    try:
        result = rag.query(query, param=QueryParam(mode=mode))
        print(result)
    except Exception as e:
        print(f"Error: {e}")

    print(f"\n{'=' * 70}")

print("\nGoodbye!")