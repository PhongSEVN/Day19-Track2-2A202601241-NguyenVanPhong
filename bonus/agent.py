"""HybridMemoryAgent -- minimal POC: episodic memory (vector store) fused
with a stable user profile + recent activity (feature-store-shaped state).

Self-contained on purpose: only imports fastembed / qdrant-client / rank-bm25
(already installed for the main lab). No import from app/ or notebooks/, so
this folder can be read, run, and graded independently of the rest of the repo.

The "feature store" here is SIMULATED as an in-memory dict whose schema
mirrors app/feast_repo/feature_views.py exactly (see ARCHITECTURE.md,
decision #2):
    user_profile_features:   reading_speed_wpm, preferred_language, topic_affinity
    query_velocity_features: queries_last_hour, distinct_topics_24h
Swapping the dict for a real `feast.FeatureStore.get_online_features()` call
does not change anything below `_profile()` / `_activity_for()`.

Retrieval reuses the same Reciprocal Rank Fusion as NB2 (k=60, 1-based rank)
and the same per-tenant payload-filter pattern as NB5's `tenant_filter`
(app/filters.py), scoped to `user_id` instead of `tenant`.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # same cached model as NB1/NB2 -- no extra download
EMBED_DIM = 384
CHUNK_WORD_LIMIT = 40                     # decision #1, see ARCHITECTURE.md
RRF_K = 60                                # same default as NB2/app.search
COLLECTION = "bonus_episodic_memory"

_PROFILE_DEFAULT = {"reading_speed_wpm": 220, "preferred_language": "vi", "topic_affinity": "cloud"}
_ACTIVITY_DEFAULT = {"queries_last_hour": 0, "distinct_topics_24h": 0}


@dataclass
class Memory:
    mem_id: int
    user_id: str
    text: str
    ts: float


class HybridMemoryAgent:
    def __init__(self) -> None:
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        )
        self._memories: dict[str, list[Memory]] = {}
        self._next_id = 0
        self._profiles: dict[str, dict] = {}
        self._activity: dict[str, dict] = {}

    # ── simulated Feast online store ────────────────────────────────────
    def set_profile(self, user_id: str, **fields) -> None:
        """Seed a stable profile -- stands in for `feast materialize` output."""
        p = self._profiles.setdefault(user_id, dict(_PROFILE_DEFAULT))
        p.update(fields)

    def _profile(self, user_id: str) -> dict:
        return self._profiles.get(user_id, _PROFILE_DEFAULT)

    def _activity_for(self, user_id: str) -> dict:
        return self._activity.get(user_id, _ACTIVITY_DEFAULT)

    def _log_query(self, user_id: str) -> None:
        a = self._activity.setdefault(user_id, dict(_ACTIVITY_DEFAULT))
        a["queries_last_hour"] += 1

    # ── chunking (decision #1) ──────────────────────────────────────────
    @staticmethod
    def _chunk(text: str) -> list[str]:
        """Greedy sentence-grouping capped at CHUNK_WORD_LIMIT words/chunk."""
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s]
        chunks: list[str] = []
        cur: list[str] = []
        for s in sentences:
            cur.append(s)
            if sum(len(c.split()) for c in cur) >= CHUNK_WORD_LIMIT:
                chunks.append(" ".join(cur))
                cur = []
        if cur:
            chunks.append(" ".join(cur))
        return chunks or [text.strip()]

    # ── write path ───────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Chunk text -> embed -> upsert into the user's episodic memory."""
        chunks = self._chunk(text)
        vectors = list(self.embedder.embed(chunks))
        points = []
        for chunk, vec in zip(chunks, vectors):
            mem = Memory(self._next_id, user_id, chunk, time.time())
            self._memories.setdefault(user_id, []).append(mem)
            points.append(models.PointStruct(
                id=self._next_id,
                vector=vec.tolist(),
                payload={"user_id": user_id, "mem_id": self._next_id, "text": chunk},
            ))
            self._next_id += 1
        self.client.upsert(collection_name=COLLECTION, points=points)

    # ── read path ────────────────────────────────────────────────────────
    def _search_vector(self, query: str, user_id: str, top_k: int) -> list[int]:
        qv = next(self.embedder.embed([query])).tolist()
        qf = models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        ])
        hits = self.client.query_points(
            collection_name=COLLECTION, query=qv, query_filter=qf, limit=top_k
        ).points
        return [h.payload["mem_id"] for h in hits]

    def _search_keyword(self, query: str, user_id: str, top_k: int) -> list[int]:
        mems = self._memories.get(user_id, [])
        if not mems:
            return []
        bm25 = BM25Okapi([m.text.lower().split() for m in mems])
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(mems)), key=lambda i: -scores[i])[:top_k]
        return [mems[i].mem_id for i in ranked]

    def _hybrid_search(self, query: str, user_id: str, top_k: int = 3) -> list[Memory]:
        depth = max(top_k * 5, 10)
        vec_ids = self._search_vector(query, user_id, depth)
        kw_ids = self._search_keyword(query, user_id, depth)
        rrf: dict[int, float] = {}
        for ids in (vec_ids, kw_ids):
            for rank, mid in enumerate(ids, start=1):
                rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (RRF_K + rank)
        top_ids = [mid for mid, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]]
        by_id = {m.mem_id: m for m in self._memories.get(user_id, [])}
        return [by_id[i] for i in top_ids if i in by_id]

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features -> assembled context."""
        self._log_query(user_id)
        profile = self._profile(user_id)
        activity = self._activity_for(user_id)
        hits = self._hybrid_search(query, user_id, top_k=3)

        memory_lines = "; ".join(f"[{h.mem_id}] {h.text}" for h in hits) or "(no matching memory)"
        return (
            f"User thích chủ đề {profile['topic_affinity']}, đọc {profile['reading_speed_wpm']} wpm, "
            f"ngôn ngữ ưu tiên {profile['preferred_language']}. "
            f"Hoạt động gần đây: {activity['queries_last_hour']} query trong 1h qua. "
            f"Top memories: {memory_lines}"
        )
