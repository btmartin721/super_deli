"""Strict named-marker rules and dependency-free configuration/API boundaries."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json
from super_deli.models.marker_config import MarkerDetectorConfig, MarkerRule

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/detection"


def payload():
    return read_json(EXAMPLES / "detector.json")


def test_roundtrip_and_order():
    config = MarkerDetectorConfig.from_dict(payload())
    assert MarkerDetectorConfig.from_dict(config.to_dict()) == config
    assert tuple(r.landmark_id for r in config.landmarks) == ("red_mark", "green_mark", "blue_mark")


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"method": "anatomical"},
        {"connectivity": 6},
        {"connectivity": 8.0},
        {"min_alpha": 0},
        {"min_alpha": 256},
        {"max_pixels": 0},
        {"max_pixels": True},
        {"max_components": -1},
        {"landmarks": []},
        {"landmarks": {}},
        {"landmarks": [None]},
        {"unknown": 1},
    ],
)
def test_invalid_detector_fields(change):
    document = payload()
    document.update(change)
    with pytest.raises(ValueError):
        MarkerDetectorConfig.from_dict(document)


@pytest.mark.parametrize("field", ["schema_version", "method", "landmarks"])
def test_required_detector_fields(field):
    document = payload()
    del document[field]
    with pytest.raises(ValueError):
        MarkerDetectorConfig.from_dict(document)


@pytest.mark.parametrize(
    "change",
    [
        {"landmark_id": ""},
        {"landmark_id": " x"},
        {"landmark_id": "a\x00b"},
        {"rgb_lower": [True, 0, 0]},
        {"rgb_upper": [256, 90, 90]},
        {"rgb_lower": [0, 0]},
        {"rgb_lower": "red"},
        {"rgb_lower": [255, 255, 255]},
        {"min_area": 0},
        {"max_area": 8},
        {"min_area": 9.0},
        {"search_box": [0, 0, 0, 1]},
        {"search_box": [0, 0, 1, 2]},
        {"search_box": [0, False, 1, 1]},
        {"search_box": [0, 0, float("nan"), 1]},
        {"search_box": [0, 0, 10**400, 1]},
        {"search_box": [0, 1]},
        {"unknown": "x"},
    ],
)
def test_invalid_marker_fields(change):
    document = payload()
    document["landmarks"][0].update(change)
    with pytest.raises(ValueError):
        MarkerDetectorConfig.from_dict(document)


def test_duplicate_and_conflicting_identity_rules():
    config = MarkerDetectorConfig.from_dict(payload())
    first = config.landmarks[0]
    with pytest.raises(ValueError, match="Duplicate"):
        replace(config, landmarks=(first, first))
    with pytest.raises(ValueError, match="Overlapping"):
        replace(config, landmarks=(first, replace(first, landmark_id="other")))
    # Shared color is safe when normalized spatial intervals are disjoint.
    split = replace(
        config,
        landmarks=(
            replace(first, search_box=(0, 0, 0.5, 1)),
            replace(first, landmark_id="other", search_box=(0.5, 0, 1, 1)),
        ),
    )
    assert len(split.landmarks) == 2
    with pytest.raises(ValueError):
        MarkerDetectorConfig(landmarks=(first,) * 129)


def test_rgb_endpoints_are_inclusive_for_rule_conflicts():
    first = MarkerRule("a", (0, 0, 0), (10, 10, 10), 1, 10)
    second = MarkerRule("b", (10, 10, 10), (20, 20, 20), 1, 10)
    with pytest.raises(ValueError, match="Overlapping"):
        MarkerDetectorConfig(landmarks=(first, second))


def test_root_config_and_workflow_imports_remain_dependency_free():
    code = """
import sys
from super_deli import load_config
from super_deli.models.marker_config import MarkerDetectorConfig
from super_deli.workflows.landmark_detection import run_landmark_detection
load_config(sys.argv[1])
assert not {'PIL', 'numpy', 'scipy', 'snpio', 'sklearn'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(EXAMPLES / "dataset.json")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
