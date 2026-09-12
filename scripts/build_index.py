"""CLI build hybrid index (Dense + Sparse) vào Qdrant."""

from pipeline.dual_embedding_qdrant import process_dual_embedding


def main() -> None:
    """Build hybrid index vào Qdrant."""
    process_dual_embedding()


if __name__ == "__main__":
    main()
