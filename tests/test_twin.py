import numpy as np
import torch

from src.balancing import build_control
from src.config import CONFIG, Technique
from src.diff_sim import build_twin, initial_state, tensor, twin_step
from src.pack import sample_packs
from src.selector import mid_values


def setup(technique: Technique):
    packs = sample_packs(5, np.random.default_rng(11), CONFIG.cell, CONFIG.pack)
    components, frequencies = mid_values(CONFIG, packs.n_packs)
    control = build_control(
        technique,
        packs,
        components[technique],
        frequencies[technique],
        CONFIG,
        track_load=True,
        gate_balancing=True,
    )
    return packs, build_twin(control, technique, CONFIG)


def test_activity_gradient_is_finite_and_throttle_reduces_transfer():
    for technique in (Technique.BUCK_BOOST, Technique.FLYING_CAPACITOR):
        packs, twin = setup(technique)
        activity = torch.ones(packs.n_packs, packs.soc0.shape[1] - 1, dtype=torch.float64)
        activity.requires_grad_(True)
        state = twin_step(twin, initial_state(twin, tensor(packs.soc0)), None, activity)
        state.terminal_v.sum().backward()
        assert torch.isfinite(activity.grad).all()
        half = twin_step(
            twin, initial_state(twin, tensor(packs.soc0)), None, 0.5 * torch.ones_like(activity)
        )
        assert torch.all(half.removed_j <= state.removed_j + 1e-12)


def test_halted_packs_do_not_move():
    packs, twin = setup(Technique.BUCK_BOOST)
    state = initial_state(twin, tensor(packs.soc0))
    halt = torch.ones(packs.n_packs, dtype=torch.bool)
    following = twin_step(twin, state, tensor(np.full(packs.n_packs, 1.0)), None, halt)
    expected = tensor(packs.soc0) - 1.0 * twin.per_amp
    assert torch.allclose(following.soc, expected)
