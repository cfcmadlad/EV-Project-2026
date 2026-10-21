import numpy as np
import pytest

from src.params import FlyingCapModel, FlyingCapSpec
from src.switched_cap import element_resistance_ohm, link_resistance_ohm, two_phase_resistance_ohm


def periodic_current_a(
    delta_v: float, c_farad: float, f_hz: float, r_one: float, r_two: float
) -> float:
    half = 0.5 / f_hz
    decay_one = np.exp(-half / (r_one * c_farad))
    decay_two = np.exp(-half / (r_two * c_farad))
    v_cap, charge = 0.0, 0.0
    for _ in range(4000):
        after_one = delta_v + (v_cap - delta_v) * decay_one
        charge = c_farad * (after_one - v_cap)
        v_cap = after_one * decay_two
    return charge * f_hz


@pytest.mark.parametrize(
    ("c_farad", "f_hz", "r_one", "r_two"),
    [
        (5.5e-3, 5.5e3, 0.0366, 0.0366),
        (1e-3, 1e3, 0.0366, 0.0366),
        (10e-3, 10e3, 0.02, 0.05),
        (2e-3, 3e3, 0.3, 0.01),
    ],
)
def test_resistance_matches_periodic_steady_state(
    c_farad: float, f_hz: float, r_one: float, r_two: float
) -> None:
    inverse_fc = np.array([1.0 / (f_hz * c_farad)])
    r_eq = two_phase_resistance_ohm(inverse_fc, np.array([r_one]), np.array([r_two]))[0]
    current = periodic_current_a(1.0, c_farad, f_hz, r_one, r_two)
    assert current == pytest.approx(1.0 / r_eq, rel=1e-9)


def test_equal_phases_reduce_to_the_coth_form() -> None:
    c_farad, f_hz, r_ohm = 5.5e-3, 5.5e3, 0.0366
    inverse_fc = np.array([1.0 / (f_hz * c_farad)])
    r_eq = two_phase_resistance_ohm(inverse_fc, np.array([r_ohm]), np.array([r_ohm]))[0]
    expected = 1.0 / (f_hz * c_farad * np.tanh(1.0 / (4.0 * f_hz * r_ohm * c_farad)))
    assert r_eq == pytest.approx(expected, rel=1e-12)


def test_slow_and_fast_limits() -> None:
    r_ohm = np.array([0.03])
    slow = two_phase_resistance_ohm(np.array([1.0 / (1e-3 * 1e3)]), r_ohm, r_ohm)[0]
    fast = two_phase_resistance_ohm(np.array([1.0 / (1.0 * 1e6)]), r_ohm, r_ohm)[0]
    assert slow == pytest.approx(1.0 / (1e-3 * 1e3), rel=1e-6)
    assert fast == pytest.approx(4.0 * r_ohm[0], rel=1e-4)


def test_phase_order_does_not_matter() -> None:
    left = two_phase_resistance_ohm(np.array([0.05]), np.array([0.02]), np.array([0.07]))
    right = two_phase_resistance_ohm(np.array([0.05]), np.array([0.07]), np.array([0.02]))
    assert left[0] == pytest.approx(right[0], rel=1e-14)


def test_element_and_cell_resistance_add_up_to_the_link() -> None:
    r0 = np.array([[0.010, 0.012, 0.014, 0.011]])
    inverse_fc = np.array([0.04])
    for model in FlyingCapModel:
        hardware = FlyingCapSpec(model=model)
        total = link_resistance_ohm(inverse_fc, r0, hardware)
        element = element_resistance_ohm(inverse_fc, r0, hardware)
        np.testing.assert_allclose(element + r0[:, :-1] + r0[:, 1:], total, rtol=1e-12)
        assert np.all(element > 0.0)


def test_midsem_option_keeps_the_original_formula() -> None:
    r0 = np.array([[0.030, 0.033]])
    inverse_fc = np.array([0.0333])
    hardware = FlyingCapSpec(model=FlyingCapModel.MIDSEM)
    expected = 0.0333 + 2.0 * 0.01 + 0.005 + 0.030 + 0.033
    assert link_resistance_ohm(inverse_fc, r0, hardware)[0, 0] == pytest.approx(expected, rel=1e-14)


def test_fast_switching_resistance_exceeds_the_midsem_value_at_baseline() -> None:
    r0 = np.array([[0.01158, 0.01158]])
    inverse_fc = np.array([1.0 / (5.5e3 * 5.5e-3)])
    midsem = link_resistance_ohm(inverse_fc, r0, FlyingCapSpec(model=FlyingCapModel.MIDSEM))
    two_phase = link_resistance_ohm(inverse_fc, r0, FlyingCapSpec(model=FlyingCapModel.TWO_PHASE))
    assert 1.5 < two_phase[0, 0] / midsem[0, 0] < 2.0
