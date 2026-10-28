import torch

WEIGHT_RANK = 2
FLOAT_BYTES = 4


def quantize_tensor(weight: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 1:
        return torch.where(weight >= 0, 1.0, -1.0) * weight.abs().mean()
    levels = 2 ** (bits - 1) - 1
    scale = weight.abs().max() / levels
    if scale == 0:
        return weight.clone()
    return torch.clamp(torch.round(weight / scale), -levels, levels) * scale


def quantize_state(
    state: dict[str, torch.Tensor], bits: int
) -> tuple[dict[str, torch.Tensor], int]:
    out, size = {}, 0
    for name, tensor in state.items():
        if tensor.ndim >= WEIGHT_RANK:
            out[name] = quantize_tensor(tensor, bits)
            size += (tensor.numel() * bits + 7) // 8 + FLOAT_BYTES
        else:
            out[name] = tensor.clone()
            size += tensor.numel() * FLOAT_BYTES
    return out, size
