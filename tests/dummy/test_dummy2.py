from typing import Any, Callable


def test_d(dummy_gen: Callable[[Any], Any]) -> None:
    assert dummy_gen == "dummy"
