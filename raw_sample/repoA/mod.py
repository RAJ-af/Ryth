"""A small, well-documented utility module."""


def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def clamp(x: int, lo: int, hi: int) -> int:
    # keep x within [lo, hi]
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
