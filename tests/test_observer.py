"""The Yang Observer must read the lattice without any way to change it."""

import numpy as np
import pytest

from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
from membrane_arena_3x3.shields import _read_only


def test_observer_accepts_locked_arrays_and_leaves_them_identical():
    arena = MembraneArena3x3(EngineConfig(seed=3))
    state = arena.lattice.state.copy()
    reference = arena.lattice.reference.copy()
    state.flags.writeable = False
    reference.flags.writeable = False
    before = state.tobytes(), reference.tobytes()

    observation = arena.observer.observe(state, reference, 0)

    assert (state.tobytes(), reference.tobytes()) == before
    assert 0.0 < observation.min_purity <= observation.mean_purity <= 1.0


def test_observer_does_not_unlock_or_alias_the_callers_array():
    arena = MembraneArena3x3(EngineConfig(seed=3))
    state = arena.lattice.state
    arena.observer.observe(state, arena.lattice.reference, 0)
    assert state.flags.writeable  # the caller's array keeps its own flags


def test_read_only_view_blocks_writes_but_shares_memory():
    base = np.zeros((2, 2))
    view = _read_only(base)
    assert np.shares_memory(base, view)
    with pytest.raises(ValueError):
        view[0, 0] = 1.0
    assert base[0, 0] == 0.0
