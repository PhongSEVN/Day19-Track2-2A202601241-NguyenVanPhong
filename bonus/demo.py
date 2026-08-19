"""5-query demo for HybridMemoryAgent -- run with: python bonus/demo.py

No LLM call. Prints the assembled context string for each query so you can
see episodic memory (vector+BM25 RRF) and profile/activity (simulated Feast
online store) fuse together, exactly what a real LLM prompt would receive.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import HybridMemoryAgent  # noqa: E402  -- needs sys.path tweak above

USER = "u_001"

MEMORIES = [
    "Kubernetes là hệ thống điều phối container, quản lý scaling và self-healing "
    "cho workload cloud-native.",
    "Tự động mở rộng hạ tầng theo lưu lượng người dùng giúp tiết kiệm chi phí "
    "server vào giờ thấp điểm, dùng Kubernetes HPA hoặc AWS Auto Scaling Group.",
    "Bảo mật cloud cần zero-trust network, mã hoá dữ liệu at-rest và in-transit, "
    "xoay vòng khoá định kỳ.",
]

QUERIES = [
    "Tôi đã đọc gì về Kubernetes?",                        # 1. vector hit only
    "Recommend đọc gì tiếp",                                # 2. needs topic_affinity
    "Tôi đang quan tâm gì gần đây?",                         # 3. needs queries_last_hour
    "Tài liệu về tự động mở rộng hạ tầng?",                  # 4. paraphrase -> vector wins
    "Cho tôi summary cloud security",                        # 5. hybrid + profile
]


def main() -> int:
    agent = HybridMemoryAgent()
    agent.set_profile(USER, topic_affinity="cloud", reading_speed_wpm=280, preferred_language="vi")
    for text in MEMORIES:
        agent.remember(text, user_id=USER)

    for i, q in enumerate(QUERIES, 1):
        print(f"\n[{i}] Query: {q}")
        print(f"    Context: {agent.recall(q, user_id=USER)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
