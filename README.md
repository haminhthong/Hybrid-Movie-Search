# MovieScout: Two-Stage Hybrid Movie Search Engine

![CI](https://img.shields.io/badge/CI-Passing-brightgreen?logo=githubactions&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-D40000?logo=qdrant&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Ruff](https://img.shields.io/badge/Lint-Ruff-D7FF64?logo=ruff&logoColor=111111)
![Pytest](https://img.shields.io/badge/Test-Pytest-0A9EDC?logo=pytest&logoColor=white)
![Docker](https://img.shields.io/badge/Run-Docker-2496ED?logo=docker&logoColor=white)

MovieScout là công cụ tìm kiếm phim kết hợp **Hybrid Retrieval** (Dense Semantic + Sparse Lexical BM25), dung hợp thứ hạng bằng **Reciprocal Rank Fusion (RRF)** và tái xếp hạng độ chính xác cao bằng **Cross-Encoder Reranker**.

---

## 1. Bài toán (Problem)

Các hệ thống tìm kiếm phim truyền thống thường dựa vào từ khóa chính xác (Keyword Matching). Tuy nhiên, người xem phim thường không nhớ chính xác tiêu đề mà tìm kiếm bằng những mô tả ngôn ngữ tự nhiên:

> *"a movie where astronauts leave Earth and a father tries to communicate with his daughter across space and time"* *(Interstellar)*  
> *"a surreal thriller about dreams within dreams and a team planting an idea"* *(Inception)*  
> *"a melancholic story about loneliness and finding connection in a near future city"* *(Her)*  

Ngược lại, khi người dùng tìm kiếm chứa tên riêng hoặc thực thể cụ thể:

> *"Christopher Nolan space survival movie with Matthew McConaughey"*  

Phương pháp Dense Retrieval thuần túy có thể gặp hiện tượng "trôi ngữ nghĩa" (semantic drift) và bỏ sót các từ khóa thực thể quan trọng. 

**Giải pháp:** Kết hợp **Dense Retrieval** (bắt trọn ngữ nghĩa, cốt truyện) cùng **BM25 Lexical Retrieval** (bắt trọn tên diễn viên, đạo diễn, từ khóa hiếm), dung hợp bằng **RRF** và tinh chỉnh thứ hạng bằng **Cross-Encoder**.

---

## 2. Kiến trúc Retrieval 2 Giai đoạn (Two-Stage Hybrid Architecture)

```
                            User Query + Filters
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       Dense Semantic Branch                   Sparse Lexical Branch
     (all-MiniLM-L6-v2 384D)                     (FastEmbed BM25)
                 │                                       │
                 ▼                                       ▼
        Top-50 Dense Points                     Top-50 Sparse Points
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                        Reciprocal Rank Fusion (RRF)
                                     ▼
                            Top-20 Candidates
                                     ▼
                          Cross-Encoder Reranker
                      (ms-marco-MiniLM-L-6-v2)
                                     ▼
                                Top-N Movies
```

### Tại sao lại chọn Hybrid (Dense + BM25)?
- **Dense Embedding (`sentence-transformers/all-MiniLM-L6-v2`)**: Nắm bắt sự tương đồng về ngữ nghĩa cốt truyện mà không cần khớp từ ngữ chính xác.
- **BM25 Sparse Vector (`Qdrant/bm25`)**: Nắm bắt tên diễn viên, đạo diễn, địa danh, hoặc từ khóa hiếm với độ chính xác từ điển cao.
- Hai nhánh retrieval được truy vấn song song trên Qdrant để tối ưu hóa latency phản hồi.

### Tại sao dùng Reciprocal Rank Fusion (RRF)?
Điểm Cosine Similarity của Dense vector và điểm BM25 có phân phối và thang đo (scale) hoàn toàn khác nhau. Thay vì phải tinh chỉnh trọng số thủ công (score weighting), RRF chuẩn hóa dựa trên thứ hạng (ranks):

$$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)} \quad (k = 60)$$

### Tại sao cần Two-Stage Cross-Encoder Reranker?
- **Bi-Encoder (Retriever)**: Mã hóa độc lập query và document thành vector tĩnh. Rất nhanh khi tìm kiếm trên toàn bộ catalog hàng chục ngàn phim, nhưng không nắm bắt được sự tương tác sâu giữa từng từ ngữ của query và document.
- **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)**: Đưa trực tiếp cặp `(query, movie_text)` vào mạng Transformer để tính attention chéo giữa từng token. Cross-Encoder chính xác hơn đáng kể nhưng tốn kém tính toán, vì vậy chỉ được áp dụng trên **Top-20 ứng viên** sau bước RRF.

### Biểu diễn dữ liệu: 1 Movie = 1 Document
Mỗi bộ phim là một thực thể độc lập gồm: Tiêu đề, Đạo diễn, Diễn viên chính, Thể loại, Từ khóa và Tóm tắt cốt truyện. Khác với các hệ thống RAG xử lý văn bản dài, toàn bộ metadata của một bộ phim được chuẩn hóa thành một document duy nhất, giúp embedding đại diện trọn vẹn ngữ cảnh của phim mà không bị phân mảnh.

---

## 3. Đánh giá thực nghiệm (Ablation Benchmark)

Pipeline được kiểm thử thực nghiệm qua 4 cấu hình (Ablation Study) trên tập truy vấn chuẩn hóa:

- **B0 - BM25 Only**: Chỉ dùng truy hồi từ vựng BM25.
- **B1 - Dense Only**: Chỉ dùng dense semantic retrieval.
- **B2 - Hybrid RRF**: Dung hợp Dense + BM25 qua RRF (không có reranker).
- **B3 - Hybrid + Cross-Encoder**: Pipeline 2-stage hoàn chỉnh.

### Các chỉ số đo lường (IR Metrics):
- **nDCG@10**: Normalized Discounted Cumulative Gain tại top 10 (đánh giá thứ hạng với mức độ liên quan đa cấp 0–3).
- **MRR@10**: Mean Reciprocal Rank tại top 10 (đánh giá vị trí xuất hiện đầu tiên của kết quả đúng).
- **Recall@10 & Recall@50**: Tỷ lệ tìm lại được các phim liên quan trong top 10 và trong candidate pool top 50.
- **Rerank Gain / Harm**: Phân tích lỗi (Error Analysis) theo dõi số lượng query được Cross-Encoder cải thiện vị trí thứ hạng so với số lượng query bị giảm bậc.

---

## 4. Cấu trúc Repository

```
Hybrid-Movie-Search/
├── app/
│   └── api.py                  # FastAPI service (endpoints: /health, /search)
├── ui/
│   └── app.py                  # Streamlit web UI (Dark Glassmorphism)
├── pipeline/
│   ├── ingest.py               # Thu thập metadata phim từ TMDB REST API
│   ├── clean.py                # Làm sạch dữ liệu, chuẩn hóa movie documents
│   └── dual_embedding_qdrant.py # Batch encode dense + sparse, nạp vào Qdrant
├── retrieval/
│   ├── config.py               # Runtime settings và thông số retrieval
│   ├── query.py                # Chuẩn hóa truy vấn và mã hóa Dense + BM25
│   ├── store.py                # Truy cập Qdrant, thực thi truy hồi song song
│   ├── ranking.py              # Reciprocal Rank Fusion (RRF) và rank assignment
│   ├── rerank.py               # Cross-Encoder candidate reranking
│   └── search.py               # Two-stage retrieval orchestrator
├── evaluation/
│   ├── dev_queries.jsonl       # Tập truy vấn validation (Dev split)
│   ├── test_queries.jsonl      # Tập truy vấn đánh giá (Test split)
│   ├── metrics.py              # Triển khai các chỉ số nDCG, MRR, Recall
│   ├── errors.py               # Phân loại failure stage và rerank gain/harm
│   └── benchmark.py            # Script chạy benchmark so sánh B0 - B3
├── scripts/
│   ├── build_index.py          # CLI build index vào Qdrant
│   ├── evaluate_dev.py         # CLI đánh giá trên Dev set
│   └── evaluate_final.py       # CLI đánh giá trên Test set
├── tests/                      # Bộ kiểm thử tự động pytest
├── Dockerfile                  # Container packaging
├── docker-compose.yml          # Triển khai Qdrant, API và UI
└── pyproject.toml              # Quản lý dependencies và metadata dự án
```

---

## 5. Hướng dẫn cài đặt & Chạy ứng dụng

### Yêu cầu tiên quyết
- Python 3.10+
- Docker & Docker Compose (khuyến nghị)
- TMDB API Key (đăng ký miễn phí tại [themoviedb.org](https://www.themoviedb.org/documentation/api))

### Cấu hình môi trường
Tạo tệp `.env` từ mẫu `.env.example`:
```bash
cp .env.example .env
```
Cập nhật `TMDB_API_KEY` và URL kết nối Qdrant trong `.env`.

---

### Cách 1: Chạy bằng Docker Compose (Khuyến nghị)

Khởi chạy đồng thời Vector Database (Qdrant), Backend API (FastAPI) và Frontend (Streamlit):
```bash
docker compose up -d
```
- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

### Cách 2: Cài đặt và chạy cục bộ (Local Development)

1. **Cài đặt thư viện:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Khởi chạy Qdrant:**
   ```bash
   docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant:v1.13.2
   ```

3. **Thu thập dữ liệu và lập chỉ mục:**
   ```bash
   # 1. Thu thập dữ liệu từ TMDB
   python -m pipeline.ingest

   # 2. Làm sạch metadata và tạo combined search text
   python -m pipeline.clean

   # 3. Mã hóa Dense + Sparse và nạp vào Qdrant
   python -m scripts.build_index
   ```

4. **Khởi chạy Backend API:**
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Khởi chạy Giao diện Streamlit:**
   ```bash
   streamlit run ui/app.py
   ```

---

## 6. Kiểm thử & Đánh giá (Testing & Benchmark)

### Chạy kiểm thử tự động và Linting:
```bash
# Linting code bằng Ruff
ruff check .

# Chạy test suite
pytest -q
```

### Chạy Ablation Benchmark:
```bash
# Đánh giá 4 baseline B0-B3 trên Dev set
python -m scripts.evaluate_dev

# Đánh giá 4 baseline B0-B3 trên Test set
python -m scripts.evaluate_final
```
Kết quả chi tiết được lưu dưới dạng CSV tại thư mục `evaluation/reports/`.
