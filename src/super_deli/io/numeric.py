"""Finite decimal parsing shared by measurement and landmark readers."""

import math
import re

_NUMERIC = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def finite_float(text: str, location: str) -> float:
    if not _NUMERIC.fullmatch(text):
        raise ValueError(f"{location}: value must be a finite decimal number")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{location}: value is outside finite float range")
    mantissa = re.split("[eE]", text)[0]
    if value == 0 and any(c in "123456789" for c in mantissa):
        raise ValueError(f"{location}: nonzero value underflows float range")
    return value
