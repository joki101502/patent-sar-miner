"""PRD R9.10 / §17.5 — MolScribe must not spawn a process pool.

MolScribe post-processes predicted graphs through `multiprocessing.Pool`. On a
spawn platform every worker re-imports `__main__`, which re-executes the entry
script; this was observed forking ~17 times during the requirements jam, and it
reproduces here as a storm of `FileNotFoundError: .../<stdin>` tracebacks that
never terminates. Capping the pool at one worker does not help — one worker is
still one re-import. The pool has to go.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CELL = Path(__file__).parent / "fixtures" / "cells" / "compound5_structure.png"


def test_importing_the_module_does_not_load_a_model():
    """PRD §17.5 — modules with top-level side effects re-run on import."""
    import sarmine.structure.molscribe as ms

    runner = ms.MolScribeRunner()
    assert runner._model is None


def test_graph_conversion_is_patched_to_run_serially():
    import sarmine.structure.molscribe as ms

    ms.MolScribeRunner._install_serial_graph_conversion()
    from molscribe import interface

    assert getattr(interface.convert_graph_to_smiles, "_sarmine_serial", False) is True


def test_serial_graph_conversion_matches_the_pool_signature():
    """It stands in for the real function, so it must accept the same arguments."""
    import sarmine.structure.molscribe as ms

    ms.MolScribeRunner._install_serial_graph_conversion()
    from molscribe import interface

    smiles, molblocks, rate = interface.convert_graph_to_smiles(
        coords=[[[0.1, 0.1], [0.2, 0.2]]],
        symbols=[["C", "C"]],
        edges=[[[0, 1], [1, 0]]],
    )
    assert len(smiles) == 1 and len(molblocks) == 1
    assert 0.0 <= rate <= 1.0


@pytest.mark.slow
@pytest.mark.skipif(not CELL.is_file(), reason="structure-cell fixture missing")
def test_prediction_never_opens_a_process_pool(monkeypatch):
    """The regression guard: a Pool anywhere in the path fails the test."""
    import multiprocessing

    import sarmine.structure.molscribe as ms

    runner = ms.MolScribeRunner()
    if not runner.available:
        pytest.skip("slim checkpoint not built")

    def forbidden(*args, **kwargs):
        raise AssertionError("MolScribe opened a multiprocessing.Pool (PRD R9.10)")

    monkeypatch.setattr(multiprocessing, "Pool", forbidden)
    from molscribe import chemistry

    monkeypatch.setattr(chemistry.multiprocessing, "Pool", forbidden)

    try:
        results = runner.predict([CELL])
    finally:
        runner.free()

    assert len(results) == 1
    assert results[0].error is None
    assert results[0].smiles
