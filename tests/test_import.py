"""Verify the package imports cleanly."""

from __future__ import annotations


def test_import_package():
    import autobliteratus

    assert autobliteratus.__version__


def test_import_tuning_package():
    import autobliteratus.tuning  # noqa: F401
