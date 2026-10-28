from scripts.claims_base import CheckedClaim, at_most, boolean, is_true, percent


def is_false(value: object) -> bool:
    return value is False


def lt(key: str) -> str:
    return f"ltspice_validation.{key}"


def pair(text: str, key: str, limit: float, note: str = "") -> CheckedClaim:
    return CheckedClaim(text, lt(key), f"<= {limit * 100:g} %", at_most(limit), percent, note)


def ltspice_claims() -> list[CheckedClaim]:
    grid = "100 point C, f, dV grid"
    in_range = "points with I_pk below Isat = sqrt(2 * 50 uJ / L)"
    return [
        pair(
            "Midsem series-form flying-capacitor current matches LTspice with real MOSFETs",
            "flying_capacitor.real.current_error_series.max_abs",
            0.05,
            grid,
        ),
        pair(
            "Coth (two-phase) flying-capacitor current matches LTspice with real MOSFETs, "
            "gate drive and dead time",
            "flying_capacitor.real.current_error_coth.max_abs",
            0.05,
            grid,
        ),
        pair(
            "Coth flying-capacitor current matches LTspice with ideal switches at the Python "
            "Rds,on",
            "flying_capacitor.ideal.current_error_coth.max_abs",
            0.05,
        ),
        pair(
            "Averaged flying-capacitor loss matches the LTspice component loss with real parts",
            "flying_capacitor.real.loss_error_coth.max_abs",
            0.15,
            "gate drive is not in the averaged loss; worst at the smallest dV",
        ),
        pair(
            "Circuit-calibrated flying-capacitor current matches LTspice real parts",
            "flying_capacitor.calibrated.current_error.max_abs",
            0.05,
        ),
        pair(
            "Flying-capacitor effective link resistance is independent of the cell voltage "
            "difference",
            "flying_capacitor.r_eff_spread_over_voltage",
            0.01,
        ),
        pair(
            "Flying-capacitor cell R0 heating booked with the mean link current matches LTspice",
            "flying_capacitor.r0_loss_vs_mean_current_form.max_abs",
            0.15,
            "the link current is pulsed",
        ),
        pair(
            "Averaged buck-boost source current matches LTspice real parts, sync rectifier",
            "buck_boost.real_sync.in_range.source_current.max_abs",
            0.05,
            in_range,
        ),
        pair(
            "Averaged buck-boost sink current matches LTspice real parts, sync rectifier",
            "buck_boost.real_sync.in_range.sink_current.max_abs",
            0.05,
            in_range,
        ),
        pair(
            "Averaged buck-boost dissipated power matches the LTspice loss without gate drive",
            "buck_boost.real_sync.in_range.loss.max_abs",
            0.15,
            in_range,
        ),
        pair(
            "Averaged buck-boost sink current matches LTspice with ideal switches",
            "buck_boost.ideal_sync.in_range.sink_current.max_abs",
            0.05,
            in_range,
        ),
        pair(
            "Averaged buck-boost sink current matches LTspice over the whole grid",
            "buck_boost.real_sync.all_points.sink_current.max_abs",
            0.05,
            "includes peak currents of 5 to 12 A above the saturation rule",
        ),
        pair(
            "Averaged buck-boost sink current matches LTspice with a Schottky freewheel diode",
            "buck_boost.real_diode.in_range.sink_current.max_abs",
            0.05,
            "the diode forward drop is not in the averaged model",
        ),
        pair(
            "Circuit-calibrated buck-boost source current matches LTspice real parts",
            "buck_boost.calibrated_in_range.source_current.max_abs",
            0.05,
            in_range,
        ),
        pair(
            "Circuit-calibrated buck-boost sink current matches LTspice real parts",
            "buck_boost.calibrated_in_range.sink_current.max_abs",
            0.05,
            in_range,
        ),
        pair(
            "Circuit-calibrated buck-boost dissipated power (gate drive included) matches "
            "LTspice real parts",
            "buck_boost.calibrated_in_range.loss.max_abs",
            0.15,
            in_range,
        ),
        CheckedClaim(
            "The buck-boost stays in discontinuous conduction (D + D2 < 1) at every grid point "
            "including 4.2 V into 3.0 V",
            lt("buck_boost.duty_plus_d2_max"),
            "< 1",
            lambda value: float(value) < 1.0,
            lambda value: f"{float(value):.4g}",
            "measured conduction fractions, real parts",
        ),
        CheckedClaim(
            "The RMS-current R0 loss form gives a buck-boost total loss closer to LTspice than "
            "the mean-current form",
            lt("buck_boost.rms_form_closer"),
            "true",
            is_true,
            boolean,
            "the mean-current form is the default",
        ),
        pair(
            "LTspice runs are at periodic steady state (current change between the last two "
            "windows)",
            "checks.pss_change_max",
            0.001,
        ),
        pair(
            "LTspice results do not change with timestep or reltol",
            "checks.sensitivity_worst",
            0.005,
        ),
        pair(
            "Energy in minus energy out equals the summed component losses (worst run, all sweeps)",
            "energy_balance_residual_max",
            0.01,
        ),
        pair(
            "Buck-boost peak inductor current equals V D Ts / L at the check points",
            "checks.peak_current_formula_error.max_abs",
            0.05,
            "V is the source cell voltage",
        ),
        pair(
            "Flying-capacitor charge per cycle out of the source cell equals the charge into "
            "the sink cell",
            "checks.flying_cycle_charge_mismatch_max",
            0.01,
        ),
        pair(
            "Passive bleed current at 20, 60 and 100 ohm matches LTspice with a real MOSFET",
            "shunt.current_error.max_abs",
            0.01,
        ),
        CheckedClaim(
            "The uncalibrated averaged converter models stay inside the 5 % current and 15 % "
            "loss tolerances against LTspice real parts",
            lt("default_decision.default_is_calibrated"),
            "false",
            is_false,
            boolean,
            "when false the circuit-calibrated model becomes the default",
        ),
    ]
