"""Calibration separation test (Checkpoint 5) — the make-or-break scientific claim.

The whole pipeline is worthless if S4+S5 cannot separate known PET hydrolases from
non-PET serine hydrolases that share the fold and triad. This test asserts that the
positive controls (IsPETase, LCC-WT) rank strictly above every negative, and that the
S165A trap (6THS) correctly yields no triad.

Requires fpocket + fetched controls; skips otherwise. (If the science ever stops
separating, this test SHOULD fail — that is a real result, not something to mask.)
"""
from __future__ import annotations

import os
import shutil

import pytest

from proteus.calibrate import preconditions, print_precondition_report, run_calibration
from proteus.utils import load_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRUCT = os.path.join(REPO, "structures")


def _guard():
    if shutil.which("fpocket") is None:
        pytest.skip("fpocket not installed")
    needed = ["6EQE", "4EB0", "6THS", "1TCA", "1EA5", "1CRL", "1EVQ"]
    missing = [p for p in needed if not os.path.exists(os.path.join(STRUCT, f"{p}.pdb"))]
    if missing:
        pytest.skip(f"controls not fetched: {missing} — run controls/fetch_controls.py")


def test_positives_separate_from_negatives():
    _guard()
    cfg = load_config()
    res = run_calibration(cfg, STRUCT)
    v = res["verdict"]
    assert v["separated"] is True, (
        f"S4+S5 failed to separate positives from negatives "
        f"(margin={v.get('margin')}, lowest_pos={v.get('min_positive')}, "
        f"max_neg={v.get('max_negative')})")
    assert v["margin"] > 0


def test_operating_point_keeps_all_positives_with_clean_precision():
    _guard()
    cfg = load_config()
    res = run_calibration(cfg, STRUCT)
    op = res["operating_point"]
    assert op["recall_positives"] == 1.0
    # with a positive margin no negative should sit above the line
    assert op["false_positives"] == 0
    assert op["precision"] == 1.0


def test_6ths_trap_has_no_triad():
    _guard()
    cfg = load_config()
    res = run_calibration(cfg, STRUCT)
    trap = res["trap"]["LCC_ICCG"]
    assert trap["triad_found"] is False, "S165A inactivated control must yield no triad"


def test_torch_precondition_is_advisory_not_blocking():
    """torch is the S3-fold dependency; calibration (S4/S5) never imports it. In the
    GCE-fold topology the local box is torch-free by design, so a missing torch must
    surface as a warning, not block calibration. The `ok` verdict ignores optional
    checks. (No fpocket/controls needed — this exercises the precondition contract.)"""
    audit = preconditions(load_config(), STRUCT)
    torch_check = next(c for c in audit["checks"] if c["name"] == "import torch")
    assert torch_check.get("optional") is True
    # ok is decided ONLY by required checks — the optional torch check never flips it
    required_ok = all(c["ok"] for c in audit["checks"] if not c.get("optional"))
    assert audit["ok"] == required_ok


def test_precondition_report_tags_optional_as_warn(capsys):
    """A failed optional precondition prints as [WARN] (not [FAIL]) and does not stop GO."""
    audit = {"ok": True, "checks": [
        {"name": "import torch", "ok": False, "detail": "missing", "optional": True},
        {"name": "fpocket on PATH", "ok": True, "detail": "/usr/bin/fpocket", "optional": False},
    ]}
    print_precondition_report(audit)
    out = capsys.readouterr().out
    assert "[WARN] import torch" in out
    assert "[FAIL]" not in out
    assert "GO" in out
