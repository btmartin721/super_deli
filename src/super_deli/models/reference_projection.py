"""Linear-memory matched reference projection through the existing strict validator."""

from dataclasses import replace

import numpy as np

from super_deli.models.matched_preprocessing import align_tables


def aligned_reference_matrix(tables, config):
    if not tables:
        raise ValueError("Both matched reference views required")
    units = sorted(tables[0].unit_ids)
    if (
        not units
        or len(set(units)) != len(units)
        or any(set(t.unit_ids) != set(units) for t in tables)
    ):
        raise ValueError("Complete unique reference projection identities required")
    dimensions = len(config.morphology_features) + len(config.ecology_features)
    if 160 * len(units) * dimensions + 4096 * len(units) > config.max_working_bytes:
        raise ValueError("Reference projection working-memory budget exceeded")
    blocks = []
    for start in range(0, len(units), 512):
        wanted, subsets = units[start : start + 512], []
        for table in tables:
            index = {unit: i for i, unit in enumerate(table.unit_ids)}
            order = [index[unit] for unit in wanted]
            subsets.append(
                replace(
                    table,
                    unit_ids=tuple(wanted),
                    observation_ids=tuple(table.observation_ids[i] for i in order),
                    values=tuple(table.values[i] for i in order),
                )
            )
        x, manifest = align_tables(subsets, config, minimum=1)
        if manifest["unit_ids"] != wanted:
            raise ValueError("Reference projection order changed")
        blocks.append(x)
    return np.vstack(blocks), units
