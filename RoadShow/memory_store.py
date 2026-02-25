import os
import uuid
import chromadb
import logging
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoryStore")

# --- Configuration ---
CHROMA_DATA_PATH = os.environ.get("CHROMA_DATA_PATH", "/workspace/qwen/agents/roadshow/memory_db")

# --- Offline Embedding Handling ---
try:
    # Try using the mirror first
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    logger.info("Successfully initialized SentenceTransformer.")
except Exception as e:
    logger.warning(f"Could not load online embedding model: {e}. Switching to DummyEmbedding for offline mode.")
    # Fallback: A dummy class so the database still functions for metadata storage
    class DummyEmbeddingFunction:
        def __call__(self, input: List[str]):
            # Return zero vectors (384 is the size for MiniLM)
            return [[0.0] * 384 for _ in input]
    embedding_fn = DummyEmbeddingFunction()

# Initialize Chroma Client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Separate collections for events and people/places
episodes_collection = client.get_or_create_collection(
    name="episodic_memory",
    embedding_function=embedding_fn
)
entities_collection = client.get_or_create_collection(
    name="entity_memory",
    embedding_function=embedding_fn
)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def upsert_memory(memory_entry: Dict[str, Any]):
    """Stores or updates a memory."""
    mem_type = memory_entry.get("type", "episodic")
    collection = entities_collection if mem_type == "entity" else episodes_collection

    mem_id = memory_entry.get("id", str(uuid.uuid4()))
    content = memory_entry.get("content", memory_entry.get("canonical_label", ""))
    
    metadata = memory_entry.get("metadata", {})
    # Flatten metadata: Chroma only supports strings, ints, floats, or bools
    cleaned_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (list, dict)):
            cleaned_metadata[k] = str(v)
        else:
            cleaned_metadata[k] = v
            
    cleaned_metadata["label"] = memory_entry.get("canonical_label", "unknown")
    cleaned_metadata["timestamp"] = datetime.now().isoformat()

    collection.upsert(
        ids=[mem_id],
        documents=[content],
        metadatas=[cleaned_metadata]
    )
    return mem_id

def find_by_text(query_text: str, mem_type: str = "episodic", top_k: int = 3) -> List[Dict]:
    """Performs a semantic vector search based on text."""
    collection = entities_collection if mem_type == "entity" else episodes_collection
    
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )

    formatted_results = []
    if results['ids'] and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
    return formatted_results

def find_by_metadata(key: str, value: str, mem_type: str = "entity") -> List[Dict]:
    """Strict filter for specific metadata (e.g., find a person by name)."""
    collection = entities_collection if mem_type == "entity" else episodes_collection
    results = collection.get(where={key: value})

    formatted_results = []
    if results['ids']:
        for i in range(len(results['ids'])):
            formatted_results.append({
                "id": results['ids'][i],
                "content": results['documents'][i],
                "metadata": results['metadatas'][i]
            })
    return formatted_results

# ============================================================================
# ADDED FOR REASONING AGENT COMPATIBILITY
# ============================================================================

def find_by_image_embedding(image_vec: List[float], mem_type: str = "episodic", top_k: int = 3) -> List[Dict]:
    """
    REQUIRED BY REASONING AGENT. 
    Performs vector search using a pre-computed embedding (e.g. from CLIP or Qwen-VL).
    """
    collection = entities_collection if mem_type == "entity" else episodes_collection
    
    results = collection.query(
        query_embeddings=[image_vec],
        n_results=top_k
    )

    formatted_results = []
    if results['ids'] and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
    return formatted_results

# ============================================================================
# UTILITY
# ============================================================================

def get_context_for_reasoning(query_clues: List[str]) -> str:
    """Helper to build a text string of memories for the LLM prompt."""
    all_context = []
    for clue in query_clues:
        hits = find_by_text(clue, top_k=2)
        for h in hits:
            all_context.append(f"Past Event: {h['content']} (Label: {h['metadata'].get('label')})")

    return "\n".join(set(all_context))