import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics.json"
CLAIMS_PATH = ROOT / "results" / "claims.md"
TECHNIQUES = (
    ("passive", "passive shunt"),
    ("buck_boost", "inductor buck-boost"),
    ("flying_capacitor", "flying capacitor"),
)
LINK_NOTE = (
    "averaged link equations book each cell's R0 drop per link, not on the net cell current; "
    "see README"
)
HEADER = (
    "| Claim | Evidence key | Value | Threshold | Status | Note |",
    "|---|---|---|---|---|---|",
)


@dataclass(frozen=True, slots=True)
class CheckedClaim:
    text: str
    evidence: str
    threshold: str
    passes: Callable[[object], bool]
    render: Callable[[object], str]
    note: str = ""


@dataclass(frozen=True, slots=True)
class UnsupportedClaim:
    text: str
    reason: str


def percent(value: object) -> str:
    return f"{float(value) * 100.0:.4g} %"


def scientific(value: object) -> str:
    return f"{float(value):.2e}"


def boolean(value: object) -> str:
    return str(bool(value)).lower()


def at_most(limit: float) -> Callable[[object], bool]:
    return lambda value: float(value) <= limit


def is_true(value: object) -> bool:
    return value is True


def checked_claims() -> list[CheckedClaim]:
    claims = [
        CheckedClaim(
            "Under the ASSUMED parameter set the code reproduces the midsem baseline "
            "times, energies and efficiencies at their printed precision",
            "physics_checks.midsem_regression.pass",
            "all 9 values equal",
            is_true,
            boolean,
        ),
        CheckedClaim(
            "Passive balancing time follows the closed form t = max_i 3600 Q_i (R_b + R0_i + R1) "
            "integral of dSOC/OCV from SOC_min + 1 % to SOC_i (baseline and 200 random packs)",
            "physics_checks.passive_closed_form.max_relative_error",
            "<= 3 %",
            at_most(0.03),
            percent,
            "baseline: see physics_checks.passive_closed_form.baseline_relative_error",
        ),
        CheckedClaim(
            "Flying-capacitor balancing conserves charge (Ah out of source cells equals Ah "
            "into sink cells, 200 random packs)",
            "physics_checks.flying_capacitor_charge.max_relative_imbalance",
            "<= 1e-9 relative",
            at_most(1e-9),
            scientific,
        ),
    ]
    for key, label in TECHNIQUES:
        base = f"physics_checks.energy_ledger.{key}"
        claims.append(
            CheckedClaim(
                f"Energy ledger closes for {label}: abs((E_removed - E_delivered) - E_ledger) "
                "/ E_removed over 200 random packs, net cell currents",
                f"{base}.max_net_vs_removed",
                "<= 1 % of E_removed",
                at_most(0.01),
                percent,
            )
        )
    for key, label in TECHNIQUES:
        base = f"physics_checks.energy_ledger.{key}"
        claims.append(
            CheckedClaim(
                f"Reported dissipated energy for {label} agrees with the net-current loss "
                "ledger to within 1 % of the dissipated energy (stricter diagnostic)",
                f"{base}.max_net_vs_dissipated",
                "<= 1 % of E_dissipated",
                at_most(0.01),
                percent,
                LINK_NOTE if key != "passive" else "bleed current is the net cell current",
            )
        )
    for key, label in TECHNIQUES:
        base = f"physics_checks.energy_ledger.{key}"
        claims.append(
            CheckedClaim(
                f"Model-internal consistency for {label}: dissipated energy equals the ledger "
                "with R0 booked per link as the averaged model does",
                f"{base}.max_per_link_vs_dissipated",
                "<= 0.1 % of E_dissipated",
                at_most(0.001),
                percent,
                "residual is first order in dt (halves when dt halves)",
            )
        )
    return claims


UNSUPPORTED = (
    UnsupportedClaim(
        "The cell parameters (Q, R0, R1, C1, OCV table) describe a real 18650 cell",
        "ASSUMED values; no fit or validation against measured data yet (Phase 1)",
    ),
    UnsupportedClaim(
        "Cell-to-cell variation (Q +/-5 %, R0 +/-10 %) reflects real packs",
        "ASSUMED; the planned dataset has a single cell, so this stays assumed",
    ),
    UnsupportedClaim(
        "The averaged buck-boost and flying-capacitor current equations match "
        "switching-level behaviour",
        "not yet compared with a circuit simulator (Phase 2)",
    ),
    UnsupportedClaim(
        "The reported energy dissipated by the active schemes is physically exact",
        "per-link R0 booking differs from the net-current ledger (FAIL rows above); "
        "a coupled link solution has not been implemented",
    ),
    UnsupportedClaim(
        "The ML surrogate is more accurate than physics-based formulas",
        "no physics baselines or cross-validated comparison yet (Phase 3)",
    ),
    UnsupportedClaim(
        "The ML surrogate is a fixed factor faster than simulation",
        "timings live in results/timing.json, depend on the machine and batch size, and "
        "are not compared with closed-form baselines yet (Phase 3)",
    ),
    UnsupportedClaim(
        "ML predictions are reliable outside the training range or carry calibrated uncertainty",
        "not evaluated (Phase 3)",
    ),
    UnsupportedClaim(
        "Balancing results hold when SOC is estimated rather than known exactly",
        "SOC is assumed known exactly (Phases 4 and 8)",
    ),
    UnsupportedClaim(
        "The ML model fits and runs on a microcontroller",
        "no embedded export or target measurement (Phase 5)",
    ),
    UnsupportedClaim(
        "Balancing results hold under driving or charging current",
        "only rest balancing is simulated (Phase 6)",
    ),
    UnsupportedClaim(
        "Balancing results hold at temperatures other than the parameter-set temperature",
        "no temperature dependence (Phase 7)",
    ),
    UnsupportedClaim(
        "Results extend to strings longer than 4 cells",
        "only 4-cell packs simulated (Phase 9)",
    ),
)


def lookup(metrics: dict, dotted: str) -> object | None:
    node: object = metrics
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def checked_row(claim: CheckedClaim, metrics: dict) -> str:
    value = lookup(metrics, claim.evidence)
    if value is None:
        return row(claim.text, claim.evidence, "missing", claim.threshold, "NOT RUN", claim.note)
    status = "PASS" if claim.passes(value) else "FAIL"
    return row(claim.text, claim.evidence, claim.render(value), claim.threshold, status, claim.note)


def row(*cells: str) -> str:
    return "| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |"


def render(metrics: dict) -> str:
    lines = [
        "# Claims",
        "",
        "Generated by `scripts/report_claims.py` from `results/metrics.json`. Do not edit by hand.",
        f"Parameter set used by run_all.py: `{metrics.get('parameter_set', 'unknown')}`.",
        "",
        *HEADER,
        *(checked_row(claim, metrics) for claim in checked_claims()),
        *(row(claim.text, "-", "-", "-", "NOT SUPPORTED", claim.reason) for claim in UNSUPPORTED),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    CLAIMS_PATH.write_text(render(metrics), newline="\n")
    print(f"wrote {CLAIMS_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
