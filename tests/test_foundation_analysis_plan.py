"""Fit-free declaration and routing tests over hand-authored file references."""

import itertools
import json
from pathlib import Path

import pytest

from super_deli.analysis import plan_analysis
from super_deli.cli import main
from super_deli.core.provenance import configuration_hash
from super_deli.core.schema import ConfigError
from super_deli.io.config import load_config
from super_deli.models.matched_analysis_config import MatchedAnalysisConfig

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/integrative/matched_dataset.json"
MATCHED_TABULAR = EXAMPLE.parents[1] / "matched-analysis/dataset.json"
METHODS = {
    "G": "baseline-genomic",
    "S": "cluster-shape",
    "E": "cluster-tabular",
    "GS": "analyze-matched-multimodal",
    "GE": "analyze-matched-genomic",
    "SE": "analyze-matched-multimodal",
    "GSE": "analyze-matched-multimodal",
}


@pytest.fixture
def declaration(tmp_path):
    kinds = {
        "G": ("genomic", "phylip"),
        "S": ("morphometric", "landmarks_csv"),
        "E": ("ecological", "csv"),
        "V": ("environmental", "csv"),
        "M": ("morphological", "csv"),
    }
    source = tmp_path / "dataset.json"
    specimens = [
        {"specimen_id": "u1", "locality_id": "l1", "population_id": "p"},
        {"specimen_id": "u2", "locality_id": "l1", "population_id": "p"},
        {"specimen_id": "u3", "locality_id": "l2", "population_id": "p"},
    ]
    views = []
    for name, (kind, fmt) in kinds.items():
        path = tmp_path / f"{name}.input"
        path.write_text("Planner must inspect the reference only.\n", encoding="utf-8")
        view = {
            "name": name,
            "kind": kind,
            "format": fmt,
            "level": "specimen",
            "path": path.name,
            "observations": [
                {"observation_id": f"{name}_{s['specimen_id']}", "unit_id": s["specimen_id"]}
                for s in specimens
            ],
        }
        if fmt == "csv":
            view["features"] = [{"name": "value", "dtype": "continuous", "unit": "unitless"}]
        if name == "S":
            view["morphometric_options"] = {"dimensions": 2, "landmark_ids": ["a", "b", "c"]}
        if name == "V":
            view["environmental_options"] = {
                "source": "hand-authored field log",
                "spatial_support": "grid_cell",
                "coordinate_reference_system": "EPSG:4326",
                "records": [
                    {
                        "observation_id": "V_u1",
                        "coordinates": [40, -74],
                        "source_cell_id": "cell7",
                        "sampling_event_id": "event2",
                        "time_window": ["2026-10-01", "2026-10-15"],
                    }
                ],
            }
        views.append(view)

    payload = {
        "schema_version": 2,
        "dataset_id": "declaration",
        "specimens": specimens,
        "populations": ["p"],
        "localities": ["l1", "l2"],
        "modalities": views,
    }

    def save():
        source.write_text(json.dumps(payload), encoding="utf-8")
        return source

    return payload, dict(zip(kinds, views, strict=True)), save


@pytest.mark.parametrize(
    "kinds", ["".join(group) for k in range(1, 5) for group in itertools.combinations("GSEV", k)]
)
def test_all_modalities(kinds, declaration, capsys):
    payload, views, save = declaration
    if "V" not in kinds:
        payload.update(schema_version=1, modalities=[views[key] for key in "GSEM"])
    path = save()
    plan = plan_analysis(path, modalities=tuple(reversed(kinds)), target_level="specimen")
    if kinds in METHODS:
        assert plan.status == "ready"
        assert plan.selection_capability == "implemented"
        assert plan.eligible_native_methods == (METHODS[kinds],)
    else:
        assert plan.status == "unsupported"
        assert plan.eligible_native_methods == ()
        if "E" in kinds:
            assert plan.selection_capability == "planned"
            assert plan.legacy_environment_route is None
            assert "distinct ecology and environment" in plan.reasons[0]
        else:
            assert plan.selection_capability == "partial"
            assert plan.legacy_environment_route == (
                "cluster-integrative" if len(kinds) > 1 else "cluster-tabular"
            )
    assert [v["name"] for v in plan.views] == list(kinds)
    assert plan.coverage["linked_count"] == 3
    assert plan.coverage["measurement_completeness"] == "unchecked"
    assert (
        main(
            [
                "analyze",
                "plan",
                str(path),
                "--target-level",
                "specimen",
                *itertools.chain.from_iterable(("--modality", name) for name in kinds),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == plan.to_dict()


def test_legacy_conventional_morphology_reordered_and_hashed(declaration):
    payload, views, save = declaration
    payload.update(schema_version=1, modalities=[views[key] for key in "GESM"])
    views["S"]["observations"].reverse()
    payload["specimens"].reverse()
    path = save()
    plan = plan_analysis(path, modalities=("M", "S", "E", "G"), target_level="specimen")
    assert plan.status == "ready"
    assert plan.same_cohort_singletons
    assert plan.eligible_native_methods == ("analyze-matched-multimodal",)
    assert plan.coverage["linked_unit_ids"] == ["u3", "u2", "u1"]
    assert plan.views[2]["target_observation_ids"]["u1"] == ["S_u1"]
    legacy = load_config(path).to_dict()
    assert all("environmental_options" not in view for view in legacy["modalities"])
    assert plan.dataset["configuration_sha256"] == configuration_hash(legacy)


def test_grouped_links_and_site_broadcast(declaration, capsys):
    payload, views, save = declaration
    payload.update(schema_version=1, modalities=[views[key] for key in "GE"])
    views["E"].update(
        level="locality",
        observations=[
            {"observation_id": "sample2", "unit_id": "l2"},
            {"observation_id": "sample1", "unit_id": "l1"},
        ],
    )
    path = save()
    conditional = plan_analysis(path, modalities=["E", "G"], target_level="locality")
    assert conditional.status == "conditional"
    assert "declare mean aggregation" in conditional.reasons[0]
    grouped = plan_analysis(
        path,
        modalities=["E", "G"],
        target_level="locality",
        aggregation={"G": "mean"},
    )
    assert grouped.status == "ready"
    assert not grouped.same_cohort_singletons
    assert grouped.eligible_native_methods == ("cluster-integrative",)
    assert grouped.views[0]["target_observation_ids"] == {"l1": ["G_u1", "G_u2"], "l2": ["G_u3"]}
    assert grouped.views[1]["target_observation_ids"] == {"l2": ["sample2"], "l1": ["sample1"]}
    assert (
        main(
            [
                "analyze",
                "plan",
                str(path),
                "--modality",
                "G",
                "--modality",
                "E",
                "--target-level",
                "locality",
                "--aggregate",
                "G=mean",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == grouped.to_dict()
    blocked = plan_analysis(path, modalities=["E", "G"], target_level="specimen")
    assert blocked.status == "conditional"
    assert "broadcast" in blocked.reasons[0]
    assert blocked.views[1]["linked_target_count"] == 0
    assert blocked.coverage["linked_count"] == 0
    assert {"view": "E", "target_level": "locality", "method": "cluster-tabular"} in (
        blocked.standalone_routes
    )


def test_partial_coverage_and_complete_cases(declaration):
    payload, views, save = declaration
    payload.update(schema_version=1, modalities=[views[key] for key in "GS"])
    views["S"]["observations"].pop()
    path = save()
    args = {"modalities": ["S", "G"], "target_level": "specimen"}
    strict = plan_analysis(path, **args)
    assert strict.status == "conditional"
    usable = plan_analysis(path, missing_views="complete_cases", **args)
    assert usable.coverage["exclusions"] == [{"unit_id": "u3", "missing_views": ["S"]}]
    assert usable.coverage["linked_count"] == 2
    assert usable.eligible_native_methods == ("cluster-integrative",)
    assert not usable.same_cohort_singletons


@pytest.mark.parametrize("missing_views", ["error", "complete_cases"])
def test_integration_requires_two_shared_targets(declaration, capsys, missing_views):
    payload, views, save = declaration
    payload.update(schema_version=1, modalities=[views["M"], views["E"]])
    payload["localities"].append("l3")
    for name, units in (("M", ("l1", "l2")), ("E", ("l1", "l3"))):
        views[name].update(
            level="locality",
            observations=[{"observation_id": f"{name}_{unit}", "unit_id": unit} for unit in units],
        )
    path = save()
    plan = plan_analysis(
        path, modalities=["M", "E"], target_level="locality", missing_views=missing_views
    )
    assert plan.status == "conditional"
    assert plan.selection_capability == "implemented"
    assert plan.coverage["linked_unit_ids"] == ["l1"]
    assert "Integration requires at least two shared target units." in plan.reasons
    if missing_views == "error":
        assert "Some declared target units lack selected views under missing_views=error." in (
            plan.reasons
        )
    assert plan.eligible_native_methods == ()
    assert plan.native_method_outputs == {}
    assert plan.standalone_routes == (
        {"view": "M", "target_level": "locality", "method": "cluster-tabular"},
        {"view": "E", "target_level": "locality", "method": "cluster-tabular"},
    )
    assert (
        main(
            [
                "analyze",
                "plan",
                str(path),
                "--modality",
                "M",
                "--modality",
                "E",
                "--target-level",
                "locality",
                "--missing-views",
                missing_views,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == plan.to_dict()


def test_covariate_only_environment_and_images(declaration):
    _, views, save = declaration
    views["V"]["features"][0]["role"] = "covariate"
    views["S"].update(format="image_manifest")
    del views["S"]["morphometric_options"]
    path = save()
    env = plan_analysis(path, modalities=["V"], target_level="specimen")
    assert env.status == "unsupported"
    assert env.legacy_environment_route is None
    assert env.to_dict()["views"][0]["environmental_options"]["records"][0]["source_cell_id"] == (
        "cell7"
    )
    assert any("adjustment method" in reason for reason in env.reasons)
    shape = plan_analysis(path, modalities=["S"], target_level="specimen")
    assert shape.status == "conditional"
    assert any("reviewed fixed landmarks" in reason for reason in shape.reasons)


@pytest.mark.parametrize(
    "change,match",
    [
        ("version", "unknown fields"),
        ("duplicate", "occurs 2 times"),
        ("link", "Unknown unit"),
        ("file", "not a regular file"),
        ("names", "Unknown modalities"),
        ("crs", "require CRS"),
    ],
)
def test_bad_input_exits_two(change, match, declaration, capsys):
    payload, views, save = declaration
    name = "V"
    if change == "version":
        payload["schema_version"] = 1
    elif change == "duplicate":
        views["V"]["observations"].append(views["V"]["observations"][0])
    elif change == "link":
        views["V"]["observations"][0]["unit_id"] = "missing"
    elif change == "file":
        views["V"]["path"] = "missing.csv"
    elif change == "names":
        name = "missing"
    else:
        del views["V"]["environmental_options"]["coordinate_reference_system"]
    path = save()
    with pytest.raises(ConfigError, match=match):
        plan_analysis(path, modalities=[name], target_level="specimen")
    assert (
        main(["analyze", "plan", str(path), "--modality", name, "--target-level", "specimen"]) == 2
    )
    captured = capsys.readouterr()
    assert not captured.out
    assert match in captured.err


def test_checked_in_example():
    plan = plan_analysis(EXAMPLE, modalities=("markers", "outline"), target_level="specimen")
    assert plan.eligible_native_methods == ("analyze-matched-multimodal",)
    assert plan.coverage["linked_count"] == plan.coverage["declared_count"] == 128
    assert any("require a content check" in check for check in plan.outstanding_checks)


def test_site_environment_requires_compatible_target_and_explicit_aggregation(declaration):
    _, views, save = declaration
    views["V"].update(
        level="locality",
        observations=[{"observation_id": f"V_{site}", "unit_id": site} for site in ("l1", "l2")],
    )
    views["V"]["environmental_options"].update(spatial_support="site", records=[])
    path = save()
    blocked = plan_analysis(
        path,
        modalities=["G", "V"],
        target_level="specimen",
        aggregation={"V": "mean"},
    )
    assert blocked.status == "unsupported"
    assert blocked.legacy_environment_route is None
    assert blocked.views[1]["target_observation_ids"] == {}
    assert any("broadcast" in reason for reason in blocked.reasons)
    unlinked = plan_analysis(path, modalities=["G", "V"], target_level="locality")
    assert unlinked.legacy_environment_route is None
    grouped = plan_analysis(
        path,
        modalities=["G", "V"],
        target_level="locality",
        aggregation={"G": "mean"},
    )
    assert grouped.status == "unsupported"
    assert grouped.legacy_environment_route == "cluster-integrative"
    assert grouped.eligible_native_methods == ()
    assert grouped.coverage["linked_count"] == grouped.coverage["declared_count"] == 2


def test_count_only_phenotype_can_use_integrator_at_specimen_level(declaration):
    payload, views, save = declaration
    views["M"]["features"][0]["dtype"] = "count"
    payload.update(schema_version=1, modalities=[views["G"], views["M"]])
    result = plan_analysis(save(), modalities=["G", "M"], target_level="specimen")
    assert result.status == "ready"
    assert result.eligible_native_methods == ("cluster-integrative",)
    assert not result.same_cohort_singletons


def test_shape_without_geometry_is_conditional_even_as_a_standalone(declaration):
    payload, views, save = declaration
    payload.update(schema_version=1, modalities=[views["S"]])
    del views["S"]["morphometric_options"]
    result = plan_analysis(save(), modalities=["S"], target_level="specimen")
    assert result.status == "conditional"
    assert result.standalone_routes == ()


@pytest.mark.parametrize("version,status", [(1, "ready"), (2, "conditional")])
def test_native_readiness_requires_schema_one(declaration, capsys, version, status):
    payload, views, save = declaration
    payload.update(schema_version=version, modalities=[views[key] for key in "GS"])
    path = save()
    original = path.read_bytes()
    result = plan_analysis(path, modalities=["G", "S"], target_level="specimen")
    assert result.status == status
    assert result.selection_capability == "implemented"
    assert result.coverage["linked_count"] == 3
    if version == 1:
        assert result.eligible_native_methods == ("analyze-matched-multimodal",)
        assert result.same_cohort_singletons
    else:
        assert result.eligible_native_methods == result.standalone_routes == ()
        assert not result.same_cohort_singletons
        assert result.reasons == ("Supply a schema-1 dataset declaration of the selected views.",)
    assert (
        main(
            [
                "analyze",
                "plan",
                str(path),
                "--modality",
                "G",
                "--modality",
                "S",
                "--target-level",
                "specimen",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == result.to_dict()
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "kinds,target,method",
    [
        ("S", "specimen", "cluster-shape"),
        ("GS", "specimen", "analyze-matched-multimodal"),
        ("GS", "locality", "cluster-integrative"),
    ],
)
@pytest.mark.parametrize("n_landmarks", [2, 3])
def test_native_shape_requires_three_landmarks(declaration, kinds, target, method, n_landmarks):
    payload, views, save = declaration
    payload.update(schema_version=1, modalities=[views[key] for key in kinds])
    views["S"]["morphometric_options"]["landmark_ids"] = ["a", "b", "c"][:n_landmarks]
    path = save()
    result = plan_analysis(
        path,
        modalities=list(kinds),
        target_level=target,
        aggregation=dict.fromkeys(kinds, "mean") if target == "locality" else None,
    )
    routes = {route["view"]: route["method"] for route in result.standalone_routes}
    if n_landmarks == 2:
        assert result.status == "conditional"
        assert result.eligible_native_methods == ()
        assert "S" not in routes
        assert any(
            "at least three fixed homologous landmarks" in reason for reason in result.reasons
        )
    else:
        assert result.status == "ready"
        assert result.eligible_native_methods == (method,)
        assert routes["S"] == "cluster-shape"
    if "G" in kinds:
        assert routes["G"] == "baseline-genomic"


def test_checked_in_matched_tabular_example(capsys):
    config = MatchedAnalysisConfig.from_dict(
        json.loads(MATCHED_TABULAR.with_name("analysis.json").read_text())
    )
    alignment = config.arms[0].alignment_config()
    names = [alignment.morphology, alignment.ecology]
    result = plan_analysis(MATCHED_TABULAR, modalities=names, target_level="specimen")
    assert result.status == "ready"
    assert result.coverage["linked_count"] == result.coverage["declared_count"] == 32
    assert result.eligible_native_methods == ("cluster-integrative", "analyze-matched-tabular")
    assert result.native_method_outputs["cluster-integrative"].startswith("One joint partition")
    assert "per prespecified" in result.native_method_outputs["analyze-matched-tabular"]
    assert not result.same_cohort_singletons
    assert result.coverage["measurement_completeness"] == "unchecked"
    assert any(
        "ecology_measurement_scope='individual'" in check for check in result.outstanding_checks
    )
    assert any("selected measurements are finite" in check for check in result.outstanding_checks)
    assert (
        main(
            [
                "analyze",
                "plan",
                str(MATCHED_TABULAR),
                "--modality",
                names[0],
                "--modality",
                names[1],
                "--target-level",
                "specimen",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == result.to_dict()


@pytest.mark.parametrize(
    "prerequisite", ["locality", "cohort_size", "continuous_features", "coverage"]
)
def test_matched_tabular_excludes_ineligible_designs(tmp_path, prerequisite):
    payload = json.loads(MATCHED_TABULAR.read_text())
    morphology, ecology = payload["modalities"]
    for view in payload["modalities"]:
        view["path"] = str(MATCHED_TABULAR.parent / view["path"])
    settings = {"target_level": "specimen"}
    if prerequisite == "locality":
        payload["localities"] = ["site_a", "site_b"]
        for specimen, site in zip(
            payload["specimens"], itertools.cycle(payload["localities"]), strict=False
        ):
            specimen["locality_id"] = site
        ecology.update(
            level="locality",
            observations=[
                {"observation_id": site, "unit_id": site} for site in payload["localities"]
            ],
        )
        settings.update(target_level="locality", aggregation={"morphology": "mean"})
    elif prerequisite == "cohort_size":
        payload["specimens"] = payload["specimens"][:9]
        for view in payload["modalities"]:
            view["observations"] = view["observations"][:9]
    elif prerequisite == "continuous_features":
        morphology["features"][2]["role"] = "covariate"
        morphology["features"][3]["dtype"] = "count"
    else:
        ecology["observations"].pop()
        settings["missing_views"] = "complete_cases"
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = plan_analysis(path, modalities=["morphology", "ecology"], **settings)
    assert result.status == "ready"
    assert result.eligible_native_methods == ("cluster-integrative",)
    assert "analyze-matched-tabular" not in result.native_method_outputs
    assert not result.same_cohort_singletons
