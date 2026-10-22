"""Dependency-free original-locus block-map parsing."""


def parse_blocks(value: object) -> tuple[str, ...]:
    """Parse a versioned map containing one block ID per original source locus."""
    if not isinstance(value, dict) or set(value) != {"schema_version", "block_ids"}:
        raise ValueError("Block map requires exactly schema_version and block_ids")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported block-map schema version")
    ids = value["block_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(i, str) or not i or i != i.strip() for i in ids)
    ):
        raise ValueError("block_ids must be a nonempty array of unpadded strings")
    return tuple(ids)
