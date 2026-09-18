# Evaluation contract

Thư mục này chứa judgments và code benchmark cho pipeline retrieval hiện tại.
Production search dùng `retrieval/search.py`; benchmark dùng cùng encoder, Qdrant
retrieval và Cross-Encoder để so sánh bốn pipeline trên cùng query.

Hai tập dữ liệu:

- `dev_queries.jsonl`: dùng để phân tích và điều chỉnh cấu hình.
- `test_queries.jsonl`: dùng sau khi cấu hình đã ổn định, không dùng để tuning.

Mỗi record có dạng:

```json
{
  "query_id": "theme_001",
  "query": "a melancholic movie about loneliness and human connection",
  "query_type": "theme",
  "judgments": {"152601": 3, "38": 2, "194": 1}
}
```

Relevance `3` là rất phù hợp, `2` phù hợp, `1` một phần và `0` không phù hợp.
Loader từ chối JSONL lỗi, judgment rỗng và `query_id` trùng.

Các pipeline benchmark:

- `B0_BM25`: ranking sparse.
- `B1_DENSE`: ranking dense.
- `B2_HYBRID_RRF`: hai ranking hợp nhất bằng RRF, giữ tối đa 20 kết quả.
- `B3_HYBRID_CE`: B2 được Cross-Encoder rerank.

Metric gồm `nDCG@10`, `MRR@10`, `Recall@10`, `Recall@50`, `Hit@1`, `Hit@5`.
`rerank_gain_harm.csv` ghi số item được cải thiện hoặc bị hạ thứ hạng. Latency
được đo một lần cho toàn bộ retrieval của query và được lặp lại trên bốn dòng
pipeline; đây là latency chung, không phải phép đo độc lập tuyệt đối.

Chạy benchmark:

```powershell
python -m scripts.evaluate_dev
python -m scripts.evaluate_final
```

Output được ghi vào `evaluation/reports/dev/` hoặc
`evaluation/reports/test/`, gồm `benchmark_results.csv`,
`benchmark_summary.csv` và `rerank_gain_harm.csv`. Các output này được gitignore
và không nên commit nếu Qdrant hoặc judgment không tái lập.
