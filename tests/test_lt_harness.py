import numpy as np
import pytest

from src.lt_asc import netlist_to_asc
from src.lt_circuits import (
    BB_PART,
    FC_PART,
    SCHOTTKY,
    Cells,
    Rectifier,
    Solver,
    Switch,
    buck_boost_netlist,
    flying_capacitor_netlist,
    shunt_netlist,
)
from src.lt_measure import Trace, Window, mean_of, mean_product
from src.lt_runs import timing

CELLS = Cells(3.79, 3.74, 0.0115)
SOLVER = Solver(1e-3, 1e-6)


def test_flying_capacitor_real_has_four_switches_and_drivers():
    built = flying_capacitor_netlist(CELLS, 5.5e-3, 5.5e3, Switch.REAL, 0.01, 0.005, SOLVER)
    assert len(built.devices) == 4
    assert built.netlist.count(FC_PART) == 4
    assert built.netlist.startswith("* ")
    assert ".tran" in built.netlist and "uic" in built.netlist


def test_flying_capacitor_ideal_uses_python_ron():
    built = flying_capacitor_netlist(CELLS, 5.5e-3, 5.5e3, Switch.IDEAL, 0.01, 0.005, SOLVER)
    assert FC_PART not in built.netlist
    assert built.netlist.count("Ron=0.01 ") == 4


@pytest.mark.parametrize("rectifier", list(Rectifier))
def test_buck_boost_variants(rectifier):
    built = buck_boost_netlist(CELLS, 55e-6, 30e3, 0.4, Switch.REAL, rectifier, 0.02, 0.08, SOLVER)
    assert built.netlist.count(BB_PART) == (2 if rectifier is Rectifier.SYNC else 1)
    assert (SCHOTTKY in built.netlist) is (rectifier is Rectifier.DIODE)
    assert "RL xl 0 0.08" in built.netlist


def test_shunt_netlist_is_operating_point():
    assert ".op" in shunt_netlist(3.79, 0.0115, 60.0, Switch.REAL)
    assert FC_PART not in shunt_netlist(3.79, 0.0115, 60.0, Switch.IDEAL)


def test_asc_has_one_flag_per_pin():
    built = flying_capacitor_netlist(CELLS, 5.5e-3, 5.5e3, Switch.REAL, 0.01, 0.005, SOLVER)
    asc = netlist_to_asc(built.netlist)
    components = sum(1 for line in asc.splitlines() if line.startswith("SYMBOL"))
    flags = sum(1 for line in asc.splitlines() if line.startswith("FLAG"))
    assert components == 22
    assert flags > components


def test_timing_windows_hold_whole_periods():
    clock = timing(5.5e3)
    periods = clock.window_s * 5.5e3
    assert periods == pytest.approx(round(periods))
    assert clock.last.start_s == pytest.approx(clock.previous.stop_s)


def test_mean_product_of_sine_squared():
    t = np.linspace(0.0, 1e-3, 20001)
    trace = Trace(t, {"v(a)": np.sin(2 * np.pi * 1e3 * t), "i(vx)": np.cos(2 * np.pi * 1e3 * t)})
    window = Window(0.0, 1e-3)
    assert mean_product(trace, window, {"V(a)": 1.0}, {"V(a)": 1.0}) == pytest.approx(0.5, rel=1e-4)
    assert mean_product(trace, window, {"V(a)": 1.0}, {"I(VX)": 1.0}) == pytest.approx(
        0.0, abs=1e-6
    )
    assert mean_of(trace, window, {"V(a)": 2.0}) == pytest.approx(0.0, abs=1e-6)
