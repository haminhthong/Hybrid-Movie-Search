"""CLI kiểm tra readiness của index hiện tại."""

from retrieval.store import check_readiness


def main() -> None:
    """In readiness contract ra stdout."""

    print(check_readiness())


if __name__ == "__main__":
    main()
