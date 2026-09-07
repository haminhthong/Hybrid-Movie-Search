"""CLI build shadow index và release alias."""

from pipeline.dual_embedding_qdrant import process_dual_embedding


def main() -> None:
    """Build và release index."""

    process_dual_embedding()


if __name__ == "__main__":
    main()
