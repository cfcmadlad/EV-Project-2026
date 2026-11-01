import numpy as np

from src.balancing import build_control
from src.config import ProjectConfig, Technique
from src.diff_sim import build_twin, initial_state, tensor, twin_step
from src.pack import sample_packs
from src.selector import mid_values
from src.stepper import PackState, cell_ocv, step

CURRENT_LIMIT_A = 3.0
RELATIVE_FLOOR = 1e-12
TECHNIQUES = (Technique.BUCK_BOOST, Technique.FLYING_CAPACITOR)


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), RELATIVE_FLOOR)))


def parity_check(config: ProjectConfig, rng: np.random.Generator) -> dict[str, object]:
    spec = config.mpc
    packs = sample_packs(spec.parity_packs, rng, config.cell, config.pack)
    currents = rng.uniform(
        -CURRENT_LIMIT_A, CURRENT_LIMIT_A, (spec.parity_steps, spec.parity_packs)
    )
    components, frequencies = mid_values(config, spec.parity_packs)
    result: dict[str, object] = {
        "packs": spec.parity_packs,
        "steps": spec.parity_steps,
        "tolerance_soc": spec.parity_tolerance_soc,
        "tolerance_energy": spec.parity_tolerance_energy,
    }
    for technique in TECHNIQUES:
        control = build_control(
            technique,
            packs,
            components[technique],
            frequencies[technique],
            config,
            track_load=True,
            gate_balancing=True,
        )
        twin = build_twin(control, technique, config)
        reference = PackState.initial(packs.soc0, cell_ocv(packs.soc0, control))
        mirror = initial_state(twin, tensor(packs.soc0))
        for k in range(spec.parity_steps):
            reference = step(reference, currents[k], control)
            mirror = twin_step(twin, mirror, tensor(currents[k]))
        errors = {
            "soc": relative(mirror.soc.numpy(), reference.soc),
            "removed_j": relative(mirror.removed_j.numpy(), reference.removed_j),
            "transferred_j": relative(mirror.transferred_j.numpy(), reference.transferred_j),
            "element_loss_j": relative(mirror.element_loss_j.numpy(), reference.element_loss_j),
            "gate_drive_j": relative(mirror.gate_drive_j.numpy(), reference.gate_drive_j),
        }
        errors["energy_max"] = max(v for k, v in errors.items() if k.endswith("_j"))
        errors["pass"] = bool(
            errors["soc"] <= spec.parity_tolerance_soc
            and errors["energy_max"] <= spec.parity_tolerance_energy
        )
        result[technique.key] = errors
    return result
