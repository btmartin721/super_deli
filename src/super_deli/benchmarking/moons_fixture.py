"""Invented, fully computed source archive for tests and installed smoke checks.

Its canonical reference identifies the numerical baseline under test; this is
explicitly a synthetic fixture, not the actual archived source study.
"""

from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import write_json


def build_fixture(output):
    from super_deli.benchmarking.density_audit import audit_density_study
    from super_deli.benchmarking.density_study import DensityStudyConfig, run_density_observation
    from super_deli.benchmarking.moons_archive import BASE_SHA
    from super_deli.models.hdbscan_config import HDBSCANConfig

    out = Path(output)
    out.mkdir()
    study = out / "study"
    study.mkdir()
    c = DensityStudyConfig(n_datasets=5, stability_datasets=5, n_subsamples=1, sample_sizes=(32,))
    m = HDBSCANConfig(True)
    from dataclasses import replace

    write_json(
        {
            "configuration": c.to_dict(),
            "phase": "main",
            "plan": c.plan("main"),
            "density_model": m.to_dict(),
            "gmm_model": replace(m.preprocessing_config(), min_cluster_size=5, n_init=10).to_dict(),
        },
        study / "frozen-plan.json",
    )
    rows = [run_density_observation((str(study), c.to_dict(), r)) for r in c.plan("main")]
    if any(a["status"] != "complete" for r in rows for a in r["families"].values()):
        raise RuntimeError("Invented source fixture fitting failed")
    write_json({"status": "complete", "synthetic_fixture": True}, study / "summary.json")
    write_json({"sha": BASE_SHA, "synthetic_fixture": True}, out / "canonical-commit.json")
    write_json({}, out / "frozen-runtime-sha256.json")
    audit_density_study(study, output_dir=out / "audit")
    return {
        "synthetic_fixture": True,
        "observations": len(rows),
        "source_fits": len(rows) * 4,
        "inventory_sha256": fingerprint_file(out / "audit/files.json").sha256,
    }
