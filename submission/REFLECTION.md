# Reflection — Lab 19

**Tên:** Nguyễn Văn Phong
**Cohort:** _A20-K3_
**Path đã chạy:** _lite_

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Kết quả không hoàn toàn giống lý thuyết trên slide. Ở query `exact`, BM25 và hybrid ngang nhau (96.7%) vì câu hỏi chứa nguyên từ khoá kỹ thuật, đúng sở trường BM25. Bất ngờ nhất là `paraphrase`: đáng lẽ vector phải thắng vì "hiểu" nghĩa, nhưng thực tế BM25 lại cao nhất (33.3%), semantic thấp nhất (24.0%), hybrid ở giữa (32.0%). Lý do: embedding model mặc định của lite path (`bge-small-en`) train chủ yếu tiếng Anh, không bắt đúng nghĩa câu tiếng Việt diễn đạt lại. Ở `mixed` thì hybrid mới cao hơn (100% so với 97–98.5%), vì đây là lúc cả hai tín hiệu đều cần cùng lúc.

Em sẽ không dùng hybrid khi: (1) query gần như chắc chắn chứa đúng keyword kỹ thuật (tra mã lỗi, ID, tên hàm) — BM25 một mình đã đủ, lại nhanh hơn nhiều (1-2ms so với vài chục ms); (2) khi embedding model không hợp ngôn ngữ của corpus như trường hợp em gặp — hybrid lúc đó chỉ kéo trung bình lên chút, không sửa được gốc vấn đề, phải đổi model mới đúng.

---

## Điều ngạc nhiên nhất khi làm lab này

Semantic search thua cả BM25 trên paraphrase - ngược hẳn với kỳ vọng ban đầu của em. Nhắc em nhớ chọn embedding model không phải chuyện nhỏ, nhất là làm việc với tiếng Việt.

---

## Bonus challenge

- [X] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _(làm một mình)_
