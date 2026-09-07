# MovieScout Hybrid Movie Retrieval

MovieScout là dịch vụ tìm kiếm phim theo query rõ ràng của người dùng. V1 tập
trung vào information retrieval có thể đo lường và tái lập:

```text
TMDB snapshot
  → validation + provenance
  → canonical movie document
  → dense + sparse embeddings
  → shadow Qdrant collection
  → validation gates + index manifest
  → atomic alias release

user query + genre/year filter
  → normalization
  → Dense + BM25 song song (top 50)
  → Reciprocal Rank Fusion (top 30)
  → Cross-Encoder rerank (top 20)
  → top N response
```

Production v1 không dùng Adaptive Router, HyDE, query rewrite hay LLM. Các
thử nghiệm này nằm trong `experiments/adaptive_search/` để so sánh riêng với
baseline, không làm tăng latency và dependency của canonical path.

## Hợp đồng dữ liệu và index

Mỗi phim là đúng một Qdrant point, với một document text versioned:

```text
Title: Interstellar. Director: Christopher Nolan. Cast: Matthew McConaughey, Anne Hathaway.
Genres: Adventure, Drama, Science Fiction. Keywords: space, wormhole.
Overview: A team of explorers travels through a wormhole in space.
```

Payload giữ metadata có cấu trúc: `genres`, `cast`, `keywords` là mảng; `director`,
`release_year`, `overview` là field riêng. Genre filter dùng categorical
exact-match, không tìm text trong chuỗi comma-separated.

Mỗi index có manifest tại `data/manifests/<index_version>.json`, gồm:

- dense model và dimension;
- sparse model và document schema version;
- dataset/index version, SHA-256 và point count;
- collection versioned và alias phục vụ.

Build luôn vào collection versioned, validate xong mới đổi alias
`movies_current`. Vì vậy index đang serve không bị trạng thái nửa cũ/nửa mới
hoặc còn movie stale khi snapshot mới nhỏ hơn snapshot cũ.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

Tạo `.env`:

```dotenv
TMDB_API_KEY=your_tmdb_api_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_INDEX_ALIAS=movies_current
INDEX_VERSION=tmdb-20260907-minilm-v1
```

`all-MiniLM-L6-v2` và benchmark v1 được định nghĩa cho query tiếng Anh.
Vietnamese/multilingual là hướng experiment riêng, không được ngầm claim là
đã hỗ trợ production.

## Chạy pipeline

```powershell
python -m pipeline.ingest
python -m pipeline.clean
python -m scripts.build_index
python -m scripts.validate_index
```

Ingestion chỉ yêu cầu overview không rỗng và metadata hợp lệ; không lọc ngầm
theo `vote_average` hoặc `popularity`. Rating/popularity chỉ là metadata hiển
thị, không được trộn vào relevance score.

Khởi động API/UI:

```powershell
uvicorn app.api:app --reload --port 8000
streamlit run ui/app_final.py
```

- `GET /health`: liveness của process.
- `GET /ready`: kiểm tra Qdrant, alias, manifest, point count và model contract.
- `POST /search`: tìm kiếm theo response contract.

Response mẫu:

```json
{
  "query": "a movie about dreams inside dreams",
  "results": [
    {
      "movie_id": 27205,
      "title": "Inception",
      "rank": 1,
      "release_year": 2010,
      "genres": ["Action", "Science Fiction"],
      "rerank_score": 6.42,
      "display_score": 1.0
    }
  ],
  "index_version": "tmdb-20260907-minilm-v1",
  "latency_ms": 182.4
}
```

`rerank_score` là điểm Cross-Encoder, không phải xác suất. `display_score` là
min-max tương đối trong result set hiện tại, không dùng để reject query, route
query hoặc so sánh giữa hai query khác nhau.

Client có thể gửi `debug=true` để nhận thêm `dense_rank`, `sparse_rank`,
`rrf_rank` và `rrf_score` trong trường `evidence`; response mặc định không lộ
các tín hiệu fusion nội bộ.

## Evaluation

`evaluation/eval_queries_200.csv` được giữ lại dưới vai trò
`synthetic_regression`: bắt regression, kiểm tra named entities/metadata và
retrieval sanity. Nó không phải headline benchmark vì nhiều query được sinh
trực tiếp từ field đang index.

Benchmark mới dùng:

- `evaluation/dev_queries.jsonl`: chọn `retrieval_k`, `candidate_k`, `rerank_k`
  và document/filter design;
- `evaluation/locked_test_queries.jsonl`: chỉ chạy sau khi config freeze;
- `evaluation/metrics.py`: `nDCG@10`, `MRR@10`, `Recall@10`, `Recall@50`,
  `Hit@1/5`;
- `evaluation/benchmark.py`: bốn ablation B0 BM25, B1 Dense, B2 Hybrid RRF,
  B3 Hybrid RRF + Cross-Encoder.

```powershell
python -m scripts.evaluate_dev
python -m scripts.evaluate_final
```

Metric chính là `nDCG@10`; known-item dùng thêm `MRR@10`, còn candidate
generation theo dõi bằng `Recall@50`. Theme query dùng graded multi-relevance
judgments (`3` rất phù hợp, `2` phù hợp, `1` một phần, `0` không phù hợp).

## Cấu trúc chính

```text
app/api.py                         FastAPI contract + liveness/readiness
pipeline/ingest.py                 TMDB snapshot + provenance
pipeline/clean.py                  canonical fields + document builder
pipeline/dual_embedding_qdrant.py shadow index + validation + alias release
pipeline/manifest.py               dataset manifest
retrieval/config.py                model/index/retrieval settings
retrieval/query.py                 query normalization + dual encoding
retrieval/store.py                 Qdrant alias + graceful branch degradation
retrieval/ranking.py               RRF + structured movie mapping
retrieval/rerank.py                Cross-Encoder candidate reranking
retrieval/search.py                canonical online orchestration
retrieval/service.py               version-aware LRU cache
retrieval/manifest.py              model--index contract
evaluation/                        judgments, metrics, benchmark, error stages
experiments/adaptive_search/       HyDE/router thử nghiệm
tests/                             unit, API và regression tests
```

## Kiểm tra chất lượng

```powershell
python -m pytest tests -q
ruff check pipeline retrieval app evaluation experiments scripts tests
```

Invariant quan trọng: movie ID duy nhất, UUID ổn định theo movie ID, document
deterministic, query/index model phải cùng contract, alias chỉ switch sau gate,
RRF deterministic, Cross-Encoder chỉ thấy fusion candidates, một nhánh Qdrant
lỗi vẫn fallback được và cả hai nhánh lỗi thì API trả `503`.
