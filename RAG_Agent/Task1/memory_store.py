# memory_store.py
import sqlite3
import json
import os
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import hashlib

# Configuration
DB_PATH = "/workspace/test/memory/memory.db"
FAISS_INDEX_PATH = "/workspace/test/memory/faiss.index"
FAISS_ID_MAP_PATH = "/workspace/test/memory/faiss_id_map.json"
DIM = 512  # chinese-clip-vit-base-patch16 dimension

def init_store():
    """Initialize SQLite database and FAISS index"""
    Path(os.path.dirname(DB_PATH)).mkdir(parents=True, exist_ok=True)
    
    # Create SQLite tables
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Main memories table
    c.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        canonical_label TEXT,
        aliases TEXT,
        metadata TEXT,
        evidence TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Image embeddings table (separate for efficient vector ops)
    c.execute("""
    CREATE TABLE IF NOT EXISTS image_embeddings (
        memory_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL,
        faiss_index INTEGER,
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    )
    """)
    
    # Index for fast lookups
    c.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_canonical_label ON memories(canonical_label)")
    
    conn.commit()
    conn.close()
    
    # Initialize FAISS index
    if not os.path.exists(FAISS_INDEX_PATH):
        index = faiss.IndexFlatL2(DIM)
        faiss.write_index(index, FAISS_INDEX_PATH)
        # Initialize empty ID mapping
        with open(FAISS_ID_MAP_PATH, 'w') as f:
            json.dump([], f)
        print(f"✓ Initialized FAISS index at {FAISS_INDEX_PATH}")
    
    print(f"✓ Memory store initialized at {DB_PATH}")


def _connect():
    return sqlite3.connect(DB_PATH)


def _load_faiss_id_map() -> List[str]:
    """Load FAISS index -> memory_id mapping"""
    if os.path.exists(FAISS_ID_MAP_PATH):
        with open(FAISS_ID_MAP_PATH, 'r') as f:
            return json.load(f)
    return []


def _save_faiss_id_map(id_map: List[str]):
    """Save FAISS index -> memory_id mapping"""
    with open(FAISS_ID_MAP_PATH, 'w') as f:
        json.dump(id_map, f)


def generate_memory_id(mem_type: str, canonical_label: str, video_context: str = "") -> str:
    """Generate deterministic memory ID for cross-video merging"""
    # Use hash of type + label for consistent IDs across videos
    content = f"{mem_type}:{canonical_label.lower().strip()}"
    hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{mem_type}_{hash_suffix}"


def upsert_memory(mem: Dict[str, Any]) -> str:
    """
    Insert or update memory. Returns memory_id.
    
    Expected mem structure:
    {
        "id": str (optional, will generate if missing),
        "type": "person_alias|place|event|object",
        "canonical_label": str,
        "aliases": [str],
        "metadata": dict,
        "image_vec": np.array (optional),
        "evidence": [str],
        "confidence": float
    }
    """
    conn = _connect()
    c = conn.cursor()
    
    # Generate ID if not provided
    if "id" not in mem or not mem["id"]:
        mem["id"] = generate_memory_id(
            mem["type"], 
            mem.get("canonical_label", "unknown"),
            mem.get("metadata", {}).get("video", "")
        )
    
    mem_id = mem["id"]
    
    # Check if memory exists
    c.execute("SELECT id, confidence, evidence, aliases FROM memories WHERE id = ?", (mem_id,))
    existing = c.fetchone()
    
    if existing:
        # Merge evidence and aliases
        old_confidence = existing[1]
        old_evidence = json.loads(existing[2] or "[]")
        old_aliases = json.loads(existing[3] or "[]")
        
        new_evidence = old_evidence + mem.get("evidence", [])
        new_aliases = list(set(old_aliases + mem.get("aliases", [])))
        new_confidence = max(old_confidence, mem.get("confidence", 0.5))
        
        # Update
        c.execute("""
            UPDATE memories 
            SET canonical_label = ?,
                aliases = ?,
                metadata = ?,
                evidence = ?,
                confidence = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            mem.get("canonical_label"),
            json.dumps(new_aliases, ensure_ascii=False),
            json.dumps(mem.get("metadata", {}), ensure_ascii=False),
            json.dumps(new_evidence, ensure_ascii=False),
            new_confidence,
            mem_id
        ))
        
        # Update image embedding if provided
        if mem.get("image_vec") is not None:
            img_blob = mem["image_vec"].astype('float32').tobytes()
            c.execute("""
                INSERT INTO image_embeddings (memory_id, embedding, faiss_index)
                VALUES (?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET embedding = excluded.embedding
            """, (mem_id, img_blob, -1))  # faiss_index updated later
    else:
        # Insert new
        c.execute("""
            INSERT INTO memories (id, type, canonical_label, aliases, metadata, evidence, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            mem_id,
            mem["type"],
            mem.get("canonical_label"),
            json.dumps(mem.get("aliases", []), ensure_ascii=False),
            json.dumps(mem.get("metadata", {}), ensure_ascii=False),
            json.dumps(mem.get("evidence", []), ensure_ascii=False),
            mem.get("confidence", 0.5)
        ))
        
        # Insert image embedding if provided
        if mem.get("image_vec") is not None:
            img_blob = mem["image_vec"].astype('float32').tobytes()
            c.execute("""
                INSERT INTO image_embeddings (memory_id, embedding, faiss_index)
                VALUES (?, ?, ?)
            """, (mem_id, img_blob, -1))
    
    conn.commit()
    conn.close()
    
    # Update FAISS index if image vector provided
    if mem.get("image_vec") is not None:
        _update_faiss_index(mem_id, mem["image_vec"])
    
    return mem_id


def _update_faiss_index(memory_id: str, vec: np.ndarray):
    """Add or update vector in FAISS index"""
    index = faiss.read_index(FAISS_INDEX_PATH)
    id_map = _load_faiss_id_map()
    
    vec_normalized = vec.astype('float32').reshape(1, -1)
    
    # Check if memory_id already in index
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT faiss_index FROM image_embeddings WHERE memory_id = ?", (memory_id,))
    result = c.fetchone()
    
    if result and result[0] >= 0:
        # Update existing vector (FAISS doesn't support update, so we rebuild)
        # For MVP, just append (TODO: periodic index rebuild for production)
        faiss_idx = len(id_map)
        index.add(vec_normalized)
        id_map.append(memory_id)
    else:
        # Add new vector
        faiss_idx = len(id_map)
        index.add(vec_normalized)
        id_map.append(memory_id)
        
        # Update faiss_index in DB
        c.execute("UPDATE image_embeddings SET faiss_index = ? WHERE memory_id = ?", 
                  (faiss_idx, memory_id))
        conn.commit()
    
    conn.close()
    
    # Save updated index and mapping
    faiss.write_index(index, FAISS_INDEX_PATH)
    _save_faiss_id_map(id_map)


def find_by_image_embedding(img_vec: np.ndarray, top_k: int = 5, 
                            threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Find similar memories by image embedding.
    Returns list of matches with similarity scores.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        return []
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    if index.ntotal == 0:
        return []
    
    id_map = _load_faiss_id_map()
    
    vec_normalized = img_vec.astype('float32').reshape(1, -1)
    D, I = index.search(vec_normalized, min(top_k, index.ntotal))
    
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(id_map):
            continue
        
        # Convert L2 distance to similarity score (0-1)
        similarity = 1.0 / (1.0 + dist)
        
        if similarity < threshold:
            continue
        
        memory_id = id_map[idx]
        mem = get_memory_by_id(memory_id)
        if mem:
            mem['similarity_score'] = float(similarity)
            mem['distance'] = float(dist)
            results.append(mem)
    
    return results


def find_by_text(query: str, mem_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find memories by text search on canonical_label and aliases"""
    conn = _connect()
    c = conn.cursor()
    
    query_lower = query.lower().strip()
    
    if mem_type:
        c.execute("""
            SELECT id, type, canonical_label, aliases, metadata, evidence, confidence
            FROM memories
            WHERE type = ? AND (
                LOWER(canonical_label) LIKE ? OR
                LOWER(aliases) LIKE ?
            )
        """, (mem_type, f"%{query_lower}%", f"%{query_lower}%"))
    else:
        c.execute("""
            SELECT id, type, canonical_label, aliases, metadata, evidence, confidence
            FROM memories
            WHERE LOWER(canonical_label) LIKE ? OR LOWER(aliases) LIKE ?
        """, (f"%{query_lower}%", f"%{query_lower}%"))
    
    results = []
    for row in c.fetchall():
        results.append({
            "id": row[0],
            "type": row[1],
            "canonical_label": row[2],
            "aliases": json.loads(row[3] or "[]"),
            "metadata": json.loads(row[4] or "{}"),
            "evidence": json.loads(row[5] or "[]"),
            "confidence": row[6]
        })
    
    conn.close()
    return results


def get_memory_by_id(memory_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single memory by ID"""
    conn = _connect()
    c = conn.cursor()
    
    c.execute("""
        SELECT id, type, canonical_label, aliases, metadata, evidence, confidence
        FROM memories WHERE id = ?
    """, (memory_id,))
    
    row = c.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "id": row[0],
        "type": row[1],
        "canonical_label": row[2],
        "aliases": json.loads(row[3] or "[]"),
        "metadata": json.loads(row[4] or "{}"),
        "evidence": json.loads(row[5] or "[]"),
        "confidence": row[6]
    }


def get_all_memories(mem_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all memories, optionally filtered by type"""
    conn = _connect()
    c = conn.cursor()
    
    if mem_type:
        c.execute("""
            SELECT id, type, canonical_label, aliases, metadata, evidence, confidence
            FROM memories WHERE type = ? ORDER BY confidence DESC
        """, (mem_type,))
    else:
        c.execute("""
            SELECT id, type, canonical_label, aliases, metadata, evidence, confidence
            FROM memories ORDER BY type, confidence DESC
        """)
    
    results = []
    for row in c.fetchall():
        results.append({
            "id": row[0],
            "type": row[1],
            "canonical_label": row[2],
            "aliases": json.loads(row[3] or "[]"),
            "metadata": json.loads(row[4] or "{}"),
            "evidence": json.loads(row[5] or "[]"),
            "confidence": row[6]
        })
    
    conn.close()
    return results

def link_place_event_entities(visual_entity_ids, video_stem, logger=print,
                              time_window_seconds: Optional[float] = None):
    """
    关联地点(place)和事件(event)，在同一视频中创建复合场景(scene)记忆。

    Args:
      visual_entity_ids: iterable of memory ids to consider
      video_stem: str, 视频标识（例如文件名不含扩展）
      logger: callable for logging
      time_window_seconds: optional float, 若提供则仅在时间重合/接近时关联
                           （要求 memories metadata 中包含 timestamp/batch 字段）
    Returns:
      list of created scene ids
    """
    logger("\n[MEMORY] Linking places and events...")
    places = []
    events = []

    for mem_id in visual_entity_ids:
        mem = get_memory_by_id(mem_id)
        if not mem:
            continue
        mem_type = mem.get('type') or ''
        if mem_type == 'place':
            places.append(mem)
        elif mem_type == 'event':
            events.append(mem)

    created_scene_ids = []
    if not places or not events:
        logger("  No place-event pairs found.")
        return created_scene_ids

    for place in places:
        for event in events:
            # 如果启用了时间窗口校验，则检查时间是否重合
            if time_window_seconds is not None:
                pt = place.get('metadata', {}).get('timestamp') or place.get('metadata', {}).get('batch')
                et = event.get('metadata', {}).get('timestamp') or event.get('metadata', {}).get('batch')
                try:
                    if pt is None or et is None:
                        # 如果任一缺失时间信息，则跳过（或可改为不跳过）
                        logger(f"  Skipping pair (missing time): {place.get('id')} ↔ {event.get('id')}")
                        continue
                    time_diff = abs(float(pt) - float(et))
                    if time_diff > float(time_window_seconds):
                        logger(f"  Skipping pair (time diff {time_diff:.1f}s > {time_window_seconds}s): "
                               f"{place.get('id')} ↔ {event.get('id')}")
                        continue
                except Exception:
                    logger("  Time check failed, skipping time filter for this pair.")
            place_label = place.get('canonical_label') or place.get('label') or f"place_{place.get('id')}"
            event_label = event.get('canonical_label') or event.get('label') or f"event_{event.get('id')}"
            scene_label = f"{place_label}_{event_label}"

            # 去重：如果同名 scene 已存在则跳过或更新（使用 find_by_text）
            existing_scenes = find_by_text(scene_label, mem_type="scene")
            if existing_scenes:
                # 更新现有 scene 的 metadata / evidence / confidence
                existing = existing_scenes[0]
                logger(f"  Scene already exists: {existing['id']}, updating metadata/evidence.")
                merged_metadata = {**existing.get('metadata', {}), "video": video_stem}
                merged_evidence = existing.get('evidence', []) + [f"compound: {place_label} during {event_label} in {video_stem}"]
                updated_mem = {
                    "id": existing['id'],
                    "type": "scene",
                    "canonical_label": existing.get('canonical_label', scene_label),
                    "metadata": merged_metadata,
                    "evidence": merged_evidence,
                    "confidence": max(existing.get('confidence', 0.5),
                                      min(float(place.get('confidence') or 0.0),
                                          float(event.get('confidence') or 0.0)) or 0.5)
                }
                upsert_memory(updated_mem)
                created_scene_ids.append(existing['id'])
                logger(f"    ✓ Updated scene memory: {existing['id']}")
                continue

            logger(f"  Linking: {place_label} ↔ {event_label}")

            compound_mem = {
                "type": "scene",
                "canonical_label": scene_label,
                "metadata": {
                    "video": video_stem,
                    "source": "compound_place_event",
                    "place_id": place.get('id'),
                    "event_id": event.get('id'),
                    "place_confidence": place.get('confidence'),
                    "event_confidence": event.get('confidence'),
                },
                "evidence": [
                    f"compound: {place_label} during {event_label} in {video_stem}"
                ],
                "confidence": min(
                    float(place.get('confidence') or 0.0),
                    float(event.get('confidence') or 0.0)
                ) or 0.5,
                "image_vec": None
            }

            scene_id = upsert_memory(compound_mem)
            created_scene_ids.append(scene_id)
            logger(f"    ✓ Created scene memory: {scene_id}")

    return created_scene_ids
    
# Initialize on import
if not os.path.exists(DB_PATH):
    init_store()