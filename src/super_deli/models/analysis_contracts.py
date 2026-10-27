"""Shared dependency-free naming and collection constraints for analysis adapters."""

import re


def validate_arm_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
        raise ValueError("Arm names must be safe lowercase identifiers of at most 64 characters")
