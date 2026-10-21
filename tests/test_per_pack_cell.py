from dataclasses import replace

import numpy as np

from src.balancing import simulate
from src.config import CONFIG, Technique
from src.experiments import baseline_pack
from src.identify import v1_response
from src.pack import tile_pack
from src.validate import rc_state


def test_explicit_uniform_cell_arrays_match_the_default_path() -> None:
    pack = tile_pack(baseline_pack(CONFIG), 3)
    explicit = replace(
        pack,
        r1_ohm=np.full(3, CONFIG.cell.r1_ohm),
        c1_farad=np.full(3, CONFIG.cell.c1_farad),
        ocv_tilt_v_per_soc=np.zeros(3),
    )
    mid = CONFIG.buck_boost
    args = (np.full(3, mid.component.mid), np.full(3, mid.frequency_hz.mid), CONFIG)
    default = simulate(Technique.BUCK_BOOST, pack, *args)
    varied = simulate(Technique.BUCK_BOOST, explicit, *args)
    assert np.array_equal(default.balancing_time_s, varied.balancing_time_s)
    assert np.allclose(default.energy_dissipated_j, varied.energy_dissipated_j, rtol=1e-12)


def test_higher_ocv_slope_speeds_up_the_flying_capacitor() -> None:
    pack = tile_pack(baseline_pack(CONFIG), 2)
    tilted = replace(pack, ocv_tilt_v_per_soc=np.array([0.0, 0.2]))
    cap = CONFIG.flying_capacitor
    args = (np.full(2, cap.component.mid), np.full(2, cap.frequency_hz.mid), CONFIG)
    times = simulate(Technique.FLYING_CAPACITOR, tilted, *args).balancing_time_s
    assert times[1] < times[0]


def test_closed_form_rc_response_matches_the_recursion() -> None:
    time_s = np.cumsum(np.full(400, 0.1))
    current = np.where(time_s < 10.0, 3.0, 0.0)
    recursion = rc_state(time_s, current, 0.02, 12.0)
    closed = v1_response(time_s, current, 0.02, 12.0)
    assert np.allclose(recursion, closed, atol=1e-12)
