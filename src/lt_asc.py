from pathlib import Path

PINS = {
    "V": ("voltage", ((0, 16), (0, 96))),
    "B": ("bv", ((0, 16), (0, 96))),
    "R": ("res", ((16, 16), (16, 96))),
    "C": ("cap", ((16, 0), (16, 64))),
    "L": ("ind", ((16, 16), (16, 96))),
    "M": ("nmos", ((48, 0), (0, 80), (48, 96))),
    "D": ("diode", ((16, 0), (16, 64))),
    "S": ("sw", ((0, 16), (0, 96), (-48, 80), (-48, 32))),
}
COLUMNS = 6
PITCH_X = 260
PITCH_Y = 260
ORIGIN = (160, 120)
SCHOTTKY_MODELS = ("MBRS340",)


def _symbol_for(kind: str, tokens: list[str]) -> str:
    if kind == "D" and tokens[-1] in SCHOTTKY_MODELS:
        return "schottky"
    return PINS[kind][0]


def _nodes(kind: str, tokens: list[str]) -> list[str]:
    count = len(PINS[kind][1])
    if kind == "M":
        return [tokens[1], tokens[2], tokens[3]]
    if kind == "S":
        return [tokens[1], tokens[2], tokens[3], "0"]
    return tokens[1 : 1 + count]


def _value(kind: str, tokens: list[str]) -> str:
    if kind in "MD":
        return tokens[-1]
    if kind == "S":
        return tokens[-1]
    return " ".join(tokens[3:])


def netlist_to_asc(netlist: str) -> str:
    lines = netlist.splitlines()
    body = [line for line in lines if line and not line.startswith("*")]
    parts = ["Version 4", "SHEET 1 1880 1200"]
    notes = []
    index = 0
    for line in body:
        tokens = line.split()
        if line.startswith("."):
            notes.append(line)
            continue
        kind = tokens[0][0].upper()
        column, row = index % COLUMNS, index // COLUMNS
        x, y = ORIGIN[0] + PITCH_X * column, ORIGIN[1] + PITCH_Y * row
        index += 1
        parts.append(f"SYMBOL {_symbol_for(kind, tokens)} {x} {y} R0")
        parts.append(f"SYMATTR InstName {tokens[0]}")
        parts.append(f"SYMATTR Value {_value(kind, tokens)}")
        for (dx, dy), node in zip(PINS[kind][1], _nodes(kind, tokens), strict=True):
            parts.append(f"FLAG {x + dx} {y + dy} {node}")
    text_y = ORIGIN[1] + PITCH_Y * (index // COLUMNS + 1)
    for note in notes:
        if note.startswith(".save"):
            continue
        parts.append(f"TEXT {ORIGIN[0]} {text_y} Left 2 !{note}")
        text_y += 40
    return "\n".join(parts) + "\n"


def write_asc(netlist: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(netlist_to_asc(netlist), newline="\n")
