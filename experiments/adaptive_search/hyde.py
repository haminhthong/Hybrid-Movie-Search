"""HyDE experimental module; không được import bởi production search."""

import logging
import re
from typing import Any

from retrieval.config import DENSE_MODEL

logger = logging.getLogger(__name__)


class HyDEProcessor:
    """Sinh hypothetical document để benchmark riêng với canonical B3."""

    def __init__(
        self,
        api_key: str | None = None,
        encoder: Any | None = None,
        embedding_model_name: str = DENSE_MODEL,
        model_name: str = "llama-3.1-8b-instant",
    ) -> None:
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(embedding_model_name)
        self.encoder = encoder
        self.model_name = model_name
        self.client: Any | None = None
        self.cache: dict[str, str] = {}
        if api_key:
            try:
                from groq import Groq

                self.client = Groq(api_key=api_key)
            except ImportError:
                logger.warning("Thiếu groq; HyDE experiment sẽ fallback về query gốc.")

    def expand(self, query: str) -> tuple[list[float], str]:
        """Sinh và encode hypothetical document, fallback nếu LLM lỗi."""

        document = self.cache.get(query, query)
        if self.client is not None and query not in self.cache:
            try:
                completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Write only a concise three-sentence movie premise."},
                        {"role": "user", "content": f"Write a hypothetical movie plot for: {query}"},
                    ],
                    model=self.model_name,
                    temperature=0.3,
                    max_tokens=150,
                )
                generated = re.sub(r"\s+", " ", completion.choices[0].message.content or "").strip()
                if len(generated) >= 20:
                    document = generated
                    self.cache[query] = generated
            except Exception:
                logger.exception("HyDE experiment lỗi; fallback về query gốc.")
        return self.encoder.encode(document, normalize_embeddings=True).tolist(), document
