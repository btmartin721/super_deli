"""Dependency-free explicit policy and lazy CLI contracts."""

from dataclasses import replace

import pytest

from super_deli.models.heldout_support_config import HeldoutSupportConfig


def test_explicit_coverage_policy():
    c = HeldoutSupportConfig()
    assert HeldoutSupportConfig.from_dict(c.to_dict()) == c
    for values in ({"schema_version": True}, {"minimum_coverage": 0.9}, {"policy": "choose_k"}):
        with pytest.raises(ValueError):
            replace(c, **values)
    for values in ({}, {**c.to_dict(), "extra": 1}):
        with pytest.raises(ValueError):
            HeldoutSupportConfig.from_dict(values)


def test_cli_is_lazy(capsys):
    from super_deli.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["assess-integrative-support", "--help"])
    assert exc.value.code == 0 and "--support-config" in capsys.readouterr().out
