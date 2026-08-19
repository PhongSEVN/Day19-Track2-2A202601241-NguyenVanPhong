# Bonus — Kiến trúc Hybrid Memory Agent

Nguyễn Văn Phong

## Sơ đồ kiến trúc

```mermaid
flowchart LR
    U[User query + user_id] --> A[HybridMemoryAgent.recall]

    subgraph Episodic["Episodic memory (Vector Store)"]
        Q[(Qdrant :memory:\ncollection bonus_episodic_memory\nfiltered by user_id)]
        B[rank-bm25\nper-user index]
    end

    subgraph Profile["Stable profile + activity (Feature-store-shaped)"]
        F[(Simulated Feast online store\nuser_profile_features\nquery_velocity_features)]
    end

    A -->|embed query| Q
    A -->|tokenize query| B
    Q -->|vector top-N| RRF{RRF fusion\nk=60, 1-based rank}
    B -->|BM25 top-N| RRF
    RRF -->|top-3 memories| C[Assembled context string]
    A -->|get_online_features-style lookup| F
    F -->|topic_affinity, reading_speed_wpm,\nqueries_last_hour| C
    C --> L[External LLM\n(not called in this POC)]

    W[HybridMemoryAgent.remember] -->|chunk -> embed -> upsert| Q
    W -->|chunk -> BM25 index| B
```

`remember()` ghi vào cả hai nhánh episodic (Qdrant + BM25 song song, giống
`app/search.py`). `recall()` truy 2 nguồn độc lập rồi ghép: episodic
(thông tin *user đã nói/đọc gì*) và profile (thông tin *user là ai*) —
đúng ranh giới vector store vs feature store của lab chính.

## 3 quyết định kiến trúc

### 1. Chunking strategy — per-message, sentence-group, cap 40 từ/chunk

`remember()` gộp câu liên tiếp tới khi đạt ~40 từ rồi cắt (`_chunk()` trong
`agent.py`). **Tradeoff:** chunk nhỏ → retrieval precision cao (mỗi fact
độc lập, tìm được đúng câu cần) nhưng tốn nhiều embed call + nhiều point
Qdrant hơn per-conversation. Chunk lớn (nguyên đoạn hội thoại) rẻ hơn về
storage nhưng vector bị pha loãng khi 1 đoạn chứa nhiều ý — đúng bài học
NB2: `semantic` mode yếu hơn `hybrid` khi 1 doc lẫn nhiều topic. Chọn
per-message vì tin nhắn hội thoại tự nhiên đã là 1 ranh giới ý — khớp với
thiết kế corpus chính (mỗi doc 1 topic).

### 2. Feature schema — tabular (named fields), không phải embedding ẩn

Profile dùng field có tên rõ (`topic_affinity`, `reading_speed_wpm`,
`preferred_language`, `queries_last_hour`, `distinct_topics_24h`) — **giống
hệt schema** `user_profile_features` + `query_velocity_features` trong
`app/feast_repo/feature_views.py` của lab chính. Lựa chọn khác là 1 vector
"latent preference" học từ history (embedding feature). **Tradeoff:**
tabular field debug được (support có thể sửa tay `topic_affinity` khi agent
đoán sai), Feast's TTL/PIT-correctness cũng thiết kế quanh field có tên,
không phải blob. Đổi lại: phải tự engineer từng signal, không "tự học" như
embedding. Chấp nhận đánh đổi vì trợ lý cá nhân cần recommendation *giải
thích được lý do*, không chỉ đúng.

### 3. Freshness strategy — 3 tier khác TTL, khớp 3 feature view của lab

- **Profile ổn định** (ngôn ngữ, tốc độ đọc) — TTL 30 ngày như
  `user_profile_features`, batch refresh hàng ngày là đủ, đổi chậm.
- **Recent activity** (`queries_last_hour`) — TTL 1 giờ như
  `query_velocity_features`, cần gần real-time (streaming Push API) để câu
  "tôi đang quan tâm gì gần đây" phản ánh đúng; batch 5 phút là ngưỡng chấp
  nhận được nếu chưa có streaming.
- **Episodic memory mới** — cần fresh gần như ngay (< 5s) vì user vừa đọc
  xong 1 tài liệu, hỏi "trợ lý nhớ gì về tôi" phải thấy ngay. POC hiện
  `remember()` upsert đồng bộ (blocking) — đủ ở scale demo, nhưng ở
  production cần queue async để không chặn request ghi.

## Lựa chọn đã cân nhắc nhưng bỏ

Tôi xem xét lưu episodic memory **trong chính Feast** (dùng embedding
feature view thay vì Qdrant riêng) để có 1 hệ thống lookup duy nhất, nhưng
bỏ vì chu kỳ refresh khác hẳn nhau: profile refresh theo batch (ngày/giờ),
memory đến liên tục theo real-time user activity (mỗi tin nhắn 1 lần ghi).
Feast's TTL/materialize cycle được thiết kế cho batch periodic, không phải
per-event ANN search — nhét episodic vào đó nghĩa là mất luôn HNSW index
của Qdrant, retrieval sẽ chậm và không scale.

## Vietnamese-context considerations

- **Code-switching vi/en** — câu như "Recommend đọc gì tiếp" rất phổ biến.
  Nhánh BM25 hiện dùng `text.lower().split()` (whitespace, giống
  `app/search.py._tokenize`) — baseline chấp nhận được cho demo nhưng
  không tách đúng từ ghép tiếng Việt đa âm tiết (vd "tự động" nên là 1
  token, không phải 2). Production cần `underthesea`/`pyvi`.
- **Nghị định 13/2023 (bảo vệ dữ liệu cá nhân VN)** — episodic memory có
  thể chứa thông tin nhạy cảm (sức khoẻ, liên hệ). POC này **không** có
  encryption at rest, không có API xoá 1 memory cụ thể (right-to-be-forgotten
  chưa đáp ứng) — ghi rõ trong Limitations.

## Limitations — What this POC doesn't handle yet

- Không mã hoá dữ liệu at-rest; Qdrant `:memory:` mất hết khi process tắt
  (đúng ý — POC, không phải production store).
- Cô lập user chỉ qua payload filter `user_id` (giống `tenant_filter` NB5),
  chưa có auth/access-control thật — 1 process compromise thấy được mọi user.
- Không có CRUD xoá/sửa 1 memory cụ thể — không đáp ứng "quyền được quên".
- Không có multi-device sync, không có memory consolidation/decay.
- BM25 tokenizer whitespace-only, yếu với từ ghép tiếng Việt.
- Feature store là dict giả lập trong process, không phải Feast thật —
  đổi sang `FeatureStore.get_online_features()` thật là drop-in, schema đã
  khớp sẵn.

## Vibe-coding note

Prompt hiệu quả nhất: yêu cầu AI viết `_hybrid_search()` bằng cách "dùng lại
đúng công thức RRF k=60 1-based rank như `app/search.py._search_hybrid`,
scope theo user_id" — AI tái dùng pattern chính xác, không cần sửa gì.
Prompt kém hiệu quả: ban đầu hỏi "thiết kế feature store cho memory agent"
không kèm ràng buộc gì — AI đề xuất 1 kiến trúc microservice quá phức tạp
cho POC 80-150 dòng; phải chỉ rõ constraint (self-contained, không phụ
thuộc app/, dùng dict giả lập) mới ra bản đúng scope.
