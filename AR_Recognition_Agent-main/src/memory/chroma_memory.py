from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document


@dataclass
class MemoryConfig:
    persist_directory: str = "chroma_db"
    collection_name: str = "ar_user_memory"


class ChromaUserMemory:
    """Simple user-scoped memory on top of Chroma.

    Stores JSON (scene summaries, user habits, interactions) as documents with metadata.
    Retrieval is filtered by `user_id`.
    """

    def __init__(self, *, vectorstore: Chroma, cfg: Optional[MemoryConfig] = None):
        self.vectorstore = vectorstore
        self.cfg = cfg or MemoryConfig()

    def upsert_event(
        self, 
        *, 
        user_id: str, 
        event: dict[str, Any], 
        text_for_embedding: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> str:
        event_id = event.get("event_id") or f"evt_{datetime.now(timezone.utc).timestamp()}"
        payload = {
            "event_id": event_id,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        
        # 合并自定义 metadata
        doc_metadata = {
            "user_id": user_id,
            "event_id": event_id,
            "created_at": payload["created_at"],
            "payload_json": json.dumps(payload, ensure_ascii=False),
        }
        if metadata:
            doc_metadata.update(metadata)
        
        doc = Document(
            page_content=text_for_embedding,
            metadata=doc_metadata,
        )
        # 使用 event_id 作为文档 ID，实现 upsert 行为
        self.vectorstore.add_documents([doc], ids=[event_id])
        return event_id

    def query(
        self, 
        *, 
        user_id: str, 
        query: str, 
        k: int = 6,
        filter_metadata: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        查询用户记忆
        
        Args:
            user_id: 用户ID
            query: 查询文本
            k: 返回结果数量
            filter_metadata: 额外的过滤条件
            
        Returns:
            包含相似度分数的结果列表
        """
        # 构建过滤器 - 使用 ChromaDB 的 $and 操作符
        filter_conditions = [{"user_id": user_id}]
        
        if filter_metadata:
            for key, value in filter_metadata.items():
                filter_conditions.append({key: value})
        
        # 如果只有一个条件，直接使用；如果多个条件，使用 $and
        if len(filter_conditions) == 1:
            filter_dict = filter_conditions[0]
        else:
            filter_dict = {"$and": filter_conditions}
        
        # 使用 similarity_search_with_score 获取相似度分数
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )
        
        out: list[dict[str, Any]] = []
        for doc, score in docs_and_scores:
            raw = doc.metadata.get("payload_json")
            if raw:
                payload = json.loads(raw)
                # 添加相似度分数 (ChromaDB 使用距离，需要转换为相似度)
                # 距离越小相似度越高，使用 1/(1+distance) 转换
                payload["similarity"] = 1.0 / (1.0 + score)
                payload["distance"] = score
                payload["metadata"] = doc.metadata
                out.append(payload)
            else:
                out.append({
                    "user_id": user_id, 
                    "event": {"text": doc.page_content},
                    "similarity": 1.0 / (1.0 + score),
                    "distance": score,
                    "metadata": doc.metadata
                })
        return out
