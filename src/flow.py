from dataclasses import dataclass

from src.config import BoolArray, FloatArray


@dataclass(frozen=True, slots=True)
class StepFlow:
    cell_current_a: FloatArray
    removed_w: FloatArray
    delivered_w: FloatArray
    link_source_a: FloatArray
    link_sink_a: FloatArray
    link_forward: BoolArray
