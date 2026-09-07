# Synthetic regression benchmark

`eval_queries_200.csv` hiện vẫn nằm ở thư mục `evaluation/` để giữ tương thích
với command legacy `retrieval.evaluate_simple`. Về mặt contract, nó thuộc split
synthetic regression: không dùng để tune hoặc làm headline benchmark.

Khi cần di chuyển vật lý file, cập nhật đồng thời đường dẫn trong
`retrieval/evaluate_simple.py` và các artifact report cũ.
