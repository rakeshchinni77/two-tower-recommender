"""Smoke test script scaffolding."""

import sys


def smoke_test(base_url: str = "http://localhost:8000") -> None:
    """Run smoke test against endpoint."""
    pass


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    smoke_test(url)
