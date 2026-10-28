from dataclasses import dataclass
from enum import StrEnum

FC_PART = "BSZ100N03LS"
BB_PART = "FDS4410"
SCHOTTKY = "MBRS340"
GATE_DRIVE_V = 5.0
GATE_RESISTOR_OHM = 2.2
EDGE_S = 10e-9
FC_DEAD_S = 0.25e-6
BB_DEAD_S = 20e-9
ZCD_THRESHOLD_A = 0.02
ROFF_OHM = 1e7
BODY_DIODE_RON_OHM = 1.0
SHUNT_OPTION = "cshunt=1e-12"
LIBRARIES = (".lib standard.mos", ".lib standard.dio")


class Switch(StrEnum):
    REAL = "real"
    IDEAL = "ideal"


class Rectifier(StrEnum):
    SYNC = "sync"
    DIODE = "diode"


@dataclass(frozen=True, slots=True)
class Cells:
    v_source: float
    v_sink: float
    r0_ohm: float


@dataclass(frozen=True, slots=True)
class Device:
    name: str
    kind: str
    plus: str
    minus: str
    current: str
    gate: str = ""
    driver_plus: str = ""
    driver_current: str = ""


@dataclass(frozen=True, slots=True)
class Built:
    netlist: str
    devices: tuple[Device, ...]


@dataclass(frozen=True, slots=True)
class Solver:
    stop_s: float
    max_step_s: float
    reltol: float = 1e-4
    abstol: float = 1e-9
    vntol: float = 1e-6


def num(value: float) -> str:
    return f"{value:.12g}"


def pulse(delay: float, width: float, period: float, level: float = GATE_DRIVE_V) -> str:
    return (
        f"PULSE(0 {num(level)} {num(delay)} {num(EDGE_S)} {num(EDGE_S)} {num(width)} {num(period)})"
    )


def _finish(
    lines: list[str], title: str, solver: Solver, extra: list[str], uic: bool, shunt: bool = False
) -> str:
    body = "\n".join([f"* {title}", *lines])
    extra = [item for item in dict.fromkeys(extra) if item != "V(0)"]
    options = (
        f".options plotwinsize=0 numdgt=15 reltol={num(solver.reltol)} "
        f"abstol={num(solver.abstol)} vntol={num(solver.vntol)}"
    )
    if shunt:
        options += f" {SHUNT_OPTION}"
    tail = [
        *LIBRARIES,
        options,
        f".save {' '.join(extra)}",
        f".tran 0 {num(solver.stop_s)} 0 {num(solver.max_step_s)}{' uic' if uic else ''}",
        ".end",
    ]
    return body + "\n" + "\n".join(tail) + "\n"


def _mos(name: str, drain: str, source: str, signal: str, part: str, driver: str = "V") -> tuple:
    lines = [f"M{name} {drain} g{name} {source} {source} {part}"]
    if driver == "V":
        lines.append(f"VD{name} gi{name} {source} {signal}")
        current = f"I(VD{name})"
    else:
        lines.append(f"BD{name} gi{name} {source} {signal}")
        current = f"I(BD{name})"
    lines.append(f"RG{name} gi{name} g{name} {num(GATE_RESISTOR_OHM)}")
    lines.append(f"RP{name} {drain} {source} {num(ROFF_OHM)}")
    device = Device(
        f"M{name}", "mos", drain, source, f"Id(M{name})", f"g{name}", f"gi{name}", current
    )
    saves = [f"Id(M{name})", f"V(g{name})", f"V(gi{name})", current, f"V({drain})", f"V({source})"]
    return lines, device, saves


def _ideal(name: str, plus: str, minus: str, control: str, ron: float) -> tuple:
    lines = [
        f"S{name} {plus} {minus} {control} 0 SW{name}",
        f".model SW{name} SW(Ron={num(ron)} Roff={num(ROFF_OHM)} Vt={num(GATE_DRIVE_V / 2)} Vh=0)",
    ]
    device = Device(f"S{name}", "switch", plus, minus, f"I(S{name})")
    return lines, device, [f"I(S{name})", f"V({plus})", f"V({minus})"]


def flying_capacitor_netlist(
    cells: Cells,
    capacitance_f: float,
    frequency_hz: float,
    switch: Switch,
    ron_ideal_ohm: float,
    esr_ohm: float,
    solver: Solver,
) -> Built:
    period = 1.0 / frequency_hz
    width = 0.5 * period - 2.0 * FC_DEAD_S - EDGE_S
    phase_one = pulse(FC_DEAD_S, width, period)
    phase_two = pulse(0.5 * period + FC_DEAD_S, width, period)
    lines = [
        f"VA ai 0 {num(cells.v_source)}",
        f"RA ai n1 {num(cells.r0_ohm)}",
        f"VB bi n1 {num(cells.v_sink)}",
        f"RB bi n2 {num(cells.r0_ohm)}",
        f"C1 cp cx {num(capacitance_f)} ic={num(0.5 * (cells.v_source + cells.v_sink))}",
        f"RE cx cm {num(esr_ohm)}",
    ]
    saves = ["I(VA)", "I(VB)", "I(RE)", "V(cp)", "V(cx)", "V(cm)", "V(n1)", "V(n2)"]
    devices = []
    layout = (
        ("1A", "cp", "n1", phase_one),
        ("1B", "cm", "0", phase_one),
        ("2A", "n2", "cp", phase_two),
        ("2B", "n1", "cm", phase_two),
    )
    for name, drain, source, signal in layout:
        if switch is Switch.REAL:
            new, device, extra = _mos(name, drain, source, signal, FC_PART)
        else:
            lines.append(f"VC{name} c{name} 0 {signal}")
            new, device, extra = _ideal(name, drain, source, f"c{name}", ron_ideal_ohm)
        lines += new
        devices.append(device)
        saves += extra
    title = f"flying capacitor {switch} C={num(capacitance_f)} f={num(frequency_hz)}"
    return Built(_finish(lines, title, solver, saves, True, True), tuple(devices))


def buck_boost_netlist(
    cells: Cells,
    inductance_h: float,
    frequency_hz: float,
    duty: float,
    switch: Switch,
    rectifier: Rectifier,
    ron_ideal_ohm: float,
    r_inductor_ohm: float,
    solver: Solver,
) -> Built:
    period = 1.0 / frequency_hz
    on_time = duty * period
    lines = [
        f"VA ai 0 {num(cells.v_source)}",
        f"RA ai a {num(cells.r0_ohm)}",
        f"VB 0 bi {num(cells.v_sink)}",
        f"RB bi bm {num(cells.r0_ohm)}",
        f"L1 x xl {num(inductance_h)}",
        f"RL xl 0 {num(r_inductor_ohm)}",
    ]
    saves = ["I(VA)", "I(VB)", "I(L1)", "V(a)", "V(x)", "V(bm)", "V(bi)"]
    dead = BB_DEAD_S if switch is Switch.REAL else 0.0
    window = f"VW w2 0 {pulse(on_time + dead, period - on_time - 2 * dead, period, 1.0)}"
    enable = f"V=5*u(V(w2)-0.5)*u(I(L1)-{num(ZCD_THRESHOLD_A)})"
    devices = []
    gate_one = pulse(0.0, on_time - EDGE_S, period)
    if switch is Switch.REAL:
        new, device, extra = _mos("1", "a", "x", gate_one, BB_PART)
        lines += new
        devices.append(device)
        saves += extra
        if rectifier is Rectifier.SYNC:
            new, device, extra = _mos("2", "x", "bm", enable, BB_PART, driver="B")
            lines += [window, *new]
            devices.append(device)
            saves += extra
        else:
            lines.append(f"D1 bm x {SCHOTTKY}")
            devices.append(Device("D1", "diode", "bm", "x", "I(D1)"))
            saves += ["I(D1)"]
    else:
        lines.append(f"VC1 c1 0 {gate_one}")
        new, device, extra = _ideal("1", "a", "x", "c1", ron_ideal_ohm)
        lines += new
        devices.append(device)
        saves += extra
        if rectifier is Rectifier.SYNC:
            lines += [window, f"BC2 c2 0 {enable}"]
            new, device, extra = _ideal("2", "bm", "x", "c2", ron_ideal_ohm)
            lines += [*new, "D2 bm x DBODY"]
            lines.append(
                f".model DBODY D(Ron={num(BODY_DIODE_RON_OHM)} Roff={num(ROFF_OHM)} Vfwd=0)"
            )
            devices += [device, Device("D2", "diode", "bm", "x", "I(D2)")]
            saves += [*extra, "I(D2)"]
        else:
            lines += [
                "D1 bm x DFW",
                f".model DFW D(Ron={num(ron_ideal_ohm)} Roff={num(ROFF_OHM)} Vfwd=0)",
            ]
            devices.append(Device("D1", "diode", "bm", "x", "I(D1)"))
            saves += ["I(D1)"]
    title = f"buck-boost {switch} {rectifier} L={num(inductance_h)} f={num(frequency_hz)}"
    return Built(_finish(lines, title, solver, saves, False), tuple(devices))


def shunt_netlist(v_cell: float, r0_ohm: float, r_bleed_ohm: float, switch: Switch) -> str:
    lines = [f"VA ai 0 {num(v_cell)}", f"RA ai a {num(r0_ohm)}"]
    if switch is Switch.REAL:
        lines += [
            f"RB a s {num(r_bleed_ohm)}",
            f"M1 s g 0 0 {FC_PART}",
            f"VG g 0 {num(GATE_DRIVE_V)}",
        ]
    else:
        lines += [f"RB a 0 {num(r_bleed_ohm)}"]
    return "\n".join([f"* shunt {switch}", *lines, *LIBRARIES, ".op", ".end"]) + "\n"
