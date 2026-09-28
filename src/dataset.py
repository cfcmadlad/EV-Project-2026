from dataclasses import dataclass

import numpy as np

from src.balancing import simulate
from src.config import BoolArray, FloatArray, IntArray, ProjectConfig, Technique
from src.pack import sample_packs

TARGET_NAMES = ("balancing_time_s", "energy_dissipated_j")


@dataclass(frozen=True)
class Dataset:
    feature_names: tuple[str, ...]
    feature_labels: tuple[str, ...]
    features: FloatArray
    targets: FloatArray
    technique: IntArray
    efficiency: FloatArray
    balanced: BoolArray


def feature_names(n_cells: int) -> tuple[str, ...]:
    cells = range(1, n_cells + 1)
    return (
        "soc_spread",
        "soc_std",
        *(f"soc0_{k}" for k in cells),
        *(f"capacity_{k}_ah" for k in cells),
        *(f"r0_{k}_ohm" for k in cells),
        *(f"is_{technique.key}" for technique in Technique),
        "component_value",
        "switching_frequency_hz",
    )


def feature_labels(n_cells: int) -> tuple[str, ...]:
    cells = range(1, n_cells + 1)
    return (
        "SOC spread",
        "SOC standard deviation",
        *(f"Initial SOC, cell {k}" for k in cells),
        *(f"Capacity, cell {k}" for k in cells),
        *(f"Resistance R0, cell {k}" for k in cells),
        *(f"Technique: {technique.label.lower()}" for technique in Technique),
        "Component value (R_b, L or C)",
        "Switching frequency",
    )


def generate_dataset(config: ProjectConfig, rng: np.random.Generator) -> Dataset:
    runs = config.dataset.runs_per_technique
    if runs < 1:
        raise ValueError(f"runs_per_technique must be positive, got {runs}")
    blocks = [_technique_block(technique, config, rng) for technique in Technique]
    features, targets, efficiency, balanced = (
        np.concatenate(parts) for parts in zip(*blocks, strict=True)
    )
    return Dataset(
        feature_names=feature_names(config.pack.n_cells),
        feature_labels=feature_labels(config.pack.n_cells),
        features=features,
        targets=targets,
        technique=np.repeat(np.arange(len(Technique)), runs),
        efficiency=efficiency,
        balanced=balanced,
    )


def _technique_block(
    technique: Technique, config: ProjectConfig, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    runs = config.dataset.runs_per_technique
    hardware = config.technique(technique)
    packs = sample_packs(runs, rng, config.cell, config.pack)
    component = rng.uniform(hardware.component.low, hardware.component.high, runs)
    frequency = rng.uniform(hardware.frequency_hz.low, hardware.frequency_hz.high, runs)
    result = simulate(technique, packs, component, frequency, config)
    one_hot = np.zeros((runs, len(Technique)))
    one_hot[:, technique] = 1.0
    features = np.column_stack(
        (
            np.ptp(packs.soc0, axis=1),
            packs.soc0.std(axis=1),
            packs.soc0,
            packs.capacity_ah,
            packs.r0_ohm,
            one_hot,
            component,
            frequency,
        )
    )
    targets = np.column_stack((result.balancing_time_s, result.energy_dissipated_j))
    return features, targets, result.efficiency, result.balanced
