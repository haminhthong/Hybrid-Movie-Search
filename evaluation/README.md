# Evaluation contract

Thư mục này tách rõ regression dữ liệu cũ khỏi benchmark dùng để kết luận chất
lượng retrieval. `eval_queries_200.csv` là **synthetic regression**: giữ lại để
bắt regression, kiểm tra entity/metadata và sanity check, không dùng làm
headline benchmark.

`dev_queries.jsonl` là tập phát triển dùng để chọn `retrieval_k`, `candidate_k`,
`rerank_k`, document schema và filter. `locked_test_queries.jsonl` chỉ chạy sau
khi cấu hình đã freeze; command final không nhận tham số tuning.

Mỗi record JSONL có schema:

```json
{
  "query_id": "theme_001",
  "query": "a melancholic movie about loneliness and human connection",
  "query_type": "theme",
  "judgments": {"152601": 3, "38": 2, "194": 1}
}
```

Relevance: `3` rất phù hợp, `2` phù hợp, `1` một phần, `0` không phù hợp.
Metric chính là `nDCG@10`; known-item dùng thêm `MRR@10`, còn candidate
generation được theo dõi bằng `Recall@50`. Benchmark B0–B3 dùng cùng một lần
encode và retrieval cho mỗi query; `latency_ms` là latency end-to-end của lần
chạy chung đó, không phải phép đo độc lập tuyệt đối của từng branch.

Kết quả benchmark được ghi vào thư mục output của command, không commit số liệu
được tạo từ Qdrant không tái lập. Cần review thủ công các judgment trước khi
dùng số liệu làm claim public.
