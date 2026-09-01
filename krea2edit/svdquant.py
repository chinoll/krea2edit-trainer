"""Experimental dual-direction SVDQuant Linear support.

The forward direction uses Nunchaku's regular W4A4 kernel.  Input gradients
are computed by running the same kernel with an independently packed,
transpose-shaped copy of the effective weight.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


# Layout constants mirror Nunchaku's Apache-2.0 NunchakuWeightPacker:
# https://github.com/nunchaku-ai/nunchaku/blob/main/nunchaku/lora/flux/packer.py
_TILE = 128
_GROUP_SIZE = 64
_N_PACKS = 8
_N_PACK_SIZE = 2
_N_LANES = 8
_K_PACK_SIZE = 2
_K_LANES = 4
_REG_K_INT4 = 8


def _pack_int4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 ``(out, in)`` values into Nunchaku's MMA layout."""
    out_features, in_features = weight.shape
    if out_features % _TILE or in_features % _TILE:
        raise ValueError("SVDQuant dimensions must both be divisible by 128")
    packed = weight.to(torch.int32).reshape(
        out_features // _TILE,
        _N_PACKS,
        _N_PACK_SIZE,
        _N_LANES,
        1,
        in_features // _GROUP_SIZE,
        1,
        _K_PACK_SIZE,
        _K_LANES,
        _REG_K_INT4,
    )
    packed = packed.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
    packed = packed.bitwise_and(0xF).bitwise_left_shift(shifts).sum(-1, dtype=torch.int32)
    return packed.view(torch.int8).view(out_features, in_features // 2)


def _unpack_int4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Undo :func:`_pack_int4_weight` and return signed INT8 values."""
    out_features, packed_in_features = weight.shape
    in_features = packed_in_features * 2
    words = weight.contiguous().view(torch.int32).reshape(
        out_features // _TILE,
        in_features // _GROUP_SIZE,
        1,
        _N_PACKS,
        _N_LANES,
        _K_LANES,
        _N_PACK_SIZE,
        _K_PACK_SIZE,
        1,
    )
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
    unpacked = words.unsqueeze(-1).bitwise_right_shift(shifts).bitwise_and(0xF)
    unpacked = unpacked.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()
    unpacked = unpacked.reshape(out_features, in_features)
    return torch.where(unpacked >= 8, unpacked - 16, unpacked).to(torch.int8)


def _pack_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pack logical per-output-channel scales shaped ``(out, groups)``."""
    out_features, groups = scale.shape
    packed = scale.reshape(out_features // _TILE, 1, 8, 2, 4, 2, groups)
    return packed.permute(0, 6, 1, 2, 4, 3, 5).contiguous().view(groups, out_features)


def _unpack_scale(scale: torch.Tensor) -> torch.Tensor:
    """Return logical scales shaped ``(out, groups)``."""
    groups, out_features = scale.shape
    unpacked = scale.contiguous().reshape(out_features // _TILE, groups, 1, 8, 4, 2, 2)
    return unpacked.permute(0, 2, 3, 5, 4, 6, 1).contiguous().view(out_features, groups)


def _pack_vector(vector: torch.Tensor) -> torch.Tensor:
    return _pack_scale(vector.reshape(-1, 1)).reshape(-1)


def _unpack_vector(vector: torch.Tensor) -> torch.Tensor:
    return _unpack_scale(vector.reshape(1, -1)).reshape(-1)


def _pack_lowrank(weight: torch.Tensor, *, down: bool) -> torch.Tensor:
    """Pack a logical Linear weight using Nunchaku's low-rank layout.

    A down projection is supplied as ``(rank, in)``.  An up projection is
    supplied as ``(out, rank)``.
    """
    pack_n = 16
    pack_k = 16
    if down:
        rank, channels = weight.shape
        rank_packs, channel_packs = rank // pack_n, channels // pack_k
        packed = weight.view(rank_packs, pack_n, channel_packs, pack_k).permute(2, 0, 1, 3)
    else:
        channels, rank = weight.shape
        channel_packs, rank_packs = channels // pack_n, rank // pack_k
        packed = weight.view(channel_packs, pack_n, rank_packs, pack_k).permute(0, 2, 1, 3)
    packed = packed.reshape(channel_packs, rank_packs, 2, 8, 1, 2, 4, 2)
    packed = packed.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
    return packed.view(channels, rank)


def _unpack_lowrank(weight: torch.Tensor, *, down: bool) -> torch.Tensor:
    channels, rank = weight.shape
    pack_n = 16
    pack_k = 16
    if down:
        rank_packs, channel_packs = rank // pack_n, channels // pack_k
    else:
        channel_packs, rank_packs = channels // pack_n, rank // pack_k
    unpacked = weight.contiguous().view(channel_packs, rank_packs, 8, 4, 2, 2, 1, 2)
    unpacked = unpacked.permute(0, 1, 4, 2, 6, 5, 3, 7).contiguous()
    unpacked = unpacked.view(channel_packs, rank_packs, pack_n, pack_k)
    if down:
        return unpacked.permute(1, 2, 0, 3).contiguous().view(rank, channels)
    return unpacked.permute(0, 2, 1, 3).contiguous().view(channels, rank)


def _source_tensor(state: dict[str, torch.Tensor], prefix: str, *names: str):
    for name in names:
        key = f"{prefix}.{name}" if prefix else name
        if key in state:
            return state[key]
    raise KeyError(f"missing {prefix}.{'/'.join(names)}")


def build_dual_svdquant_state(
    forward_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Add independently packed transpose weights to a forward INT4 checkpoint.

    Input keys follow Nunchaku's module state dict naming.  Both current Python
    names (``smooth_factor``/``proj_down``) and exported checkpoint names
    (``smooth``/``lora_down``) are accepted.
    """
    prefixes = sorted(
        key.removesuffix(".qweight")
        for key in forward_state
        if key.endswith(".qweight") and ".forward_linear." not in key
    )
    if not prefixes:
        raise ValueError("forward checkpoint contains no SVDQuant qweight tensors")
    dual_state: dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        qweight = _source_tensor(forward_state, prefix, "qweight")
        wscales = _source_tensor(forward_state, prefix, "wscales")
        smooth = _source_tensor(forward_state, prefix, "smooth_factor", "smooth")
        proj_down = _source_tensor(forward_state, prefix, "proj_down", "lora_down")
        proj_up = _source_tensor(forward_state, prefix, "proj_up", "lora_up")

        logical_qweight = _unpack_int4_weight(qweight).float()
        logical_scales = _unpack_scale(wscales).float()
        logical_smooth = _unpack_vector(smooth).float()
        main_weight = logical_qweight * logical_scales.repeat_interleave(_GROUP_SIZE, dim=1)
        main_weight.div_(logical_smooth.unsqueeze(0))

        backward_weight = main_weight.t().contiguous()
        grouped = backward_weight.view(backward_weight.shape[0], -1, _GROUP_SIZE)
        positive = grouped.amax(dim=-1)
        negative = -grouped.amin(dim=-1)
        backward_scales = torch.maximum(positive / 7.0, negative / 8.0)
        backward_scales.masked_fill_(backward_scales == 0, 1.0)
        backward_qweight = (
            grouped.div(backward_scales.unsqueeze(-1)).round_().clamp_(-8, 7).to(torch.int32)
        ).view_as(backward_weight)

        logical_down = _unpack_lowrank(proj_down, down=True)
        logical_up = _unpack_lowrank(proj_up, down=False)
        backward_down = _pack_lowrank(logical_up.t().contiguous(), down=True)
        backward_up = _pack_lowrank(logical_down.t().contiguous(), down=False)
        backward_smooth = torch.ones(
            backward_weight.shape[1], dtype=smooth.dtype, device=smooth.device
        )

        forward_prefix = f"{prefix}.forward_linear" if prefix else "forward_linear"
        backward_prefix = f"{prefix}.backward_linear" if prefix else "backward_linear"
        dual_state[f"{forward_prefix}.qweight"] = qweight
        dual_state[f"{forward_prefix}.wscales"] = wscales
        dual_state[f"{forward_prefix}.smooth_factor"] = smooth
        smooth_orig = forward_state.get(
            f"{prefix}.smooth_factor_orig", forward_state.get(f"{prefix}.smooth_orig")
        )
        dual_state[f"{forward_prefix}.smooth_factor_orig"] = (
            smooth_orig if smooth_orig is not None else smooth.clone()
        )
        dual_state[f"{forward_prefix}.proj_down"] = proj_down
        dual_state[f"{forward_prefix}.proj_up"] = proj_up
        bias_key = f"{prefix}.bias" if prefix else "bias"
        if bias_key in forward_state:
            dual_state[f"{forward_prefix}.bias"] = forward_state[bias_key]

        dual_state[f"{backward_prefix}.qweight"] = _pack_int4_weight(backward_qweight)
        dual_state[f"{backward_prefix}.wscales"] = _pack_scale(
            backward_scales.to(wscales.dtype)
        )
        packed_backward_smooth = _pack_vector(backward_smooth)
        dual_state[f"{backward_prefix}.smooth_factor"] = packed_backward_smooth
        dual_state[f"{backward_prefix}.smooth_factor_orig"] = packed_backward_smooth.clone()
        dual_state[f"{backward_prefix}.proj_down"] = backward_down
        dual_state[f"{backward_prefix}.proj_up"] = backward_up
    return dual_state


def convert_forward_svdquant_checkpoint(source: Path, output: Path):
    state = load_file(str(source), device="cpu")
    dual_state = build_dual_svdquant_state(state)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        dual_state,
        str(output),
        metadata={
            "format": "krea2edit-dual-svdquant",
            "precision": "int4",
            "backward": "transpose-w4a4-rtn",
        },
    )


def _run_linear(module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    leading_shape = tensor.shape[:-1]
    output = module(tensor.reshape(1, -1, tensor.shape[-1]).contiguous())
    return output.reshape(*leading_shape, output.shape[-1])


class _DualSVDQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, forward_linear, backward_linear):
        ctx.backward_linear = backward_linear
        return _run_linear(forward_linear, tensor)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _run_linear(ctx.backward_linear, grad_output.contiguous())
        return grad_input, None, None


class DualSVDQuantLinear(nn.Module):
    """A frozen quantized Linear with a second quantized direction for dX."""

    def __init__(self, forward_linear: nn.Module, backward_linear: nn.Module):
        super().__init__()
        self.forward_linear = forward_linear
        self.backward_linear = backward_linear
        self.in_features = forward_linear.in_features
        self.out_features = forward_linear.out_features

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return _DualSVDQuantFunction.apply(tensor, self.forward_linear, self.backward_linear)


def _make_nunchaku_linear(
    state: dict[str, torch.Tensor], prefix: str, in_features: int, out_features: int
):
    from nunchaku.models.linear import SVDQW4A4Linear

    proj_down = state[f"{prefix}.proj_down"]
    has_bias = f"{prefix}.bias" in state
    module = SVDQW4A4Linear(
        in_features=in_features,
        out_features=out_features,
        rank=proj_down.shape[1],
        bias=has_bias,
        precision="int4",
        act_unsigned=False,
        torch_dtype=proj_down.dtype,
        device="cpu",
    )
    module_state = {
        key.removeprefix(f"{prefix}."): value
        for key, value in state.items()
        if key.startswith(f"{prefix}.")
    }
    module.load_state_dict(module_state, strict=True, assign=True)
    module.requires_grad_(False)
    return module


def _set_base_layer(root: nn.Module, name: str, replacement: nn.Module):
    target = root.get_submodule(name)
    if hasattr(target, "base_layer"):
        target.base_layer = replacement
        return
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def load_dual_svdquant_linears(model: nn.Module, checkpoint: Path) -> int:
    """Replace checkpoint-covered Linear bases with dual SVDQuant modules."""
    state = load_file(str(checkpoint), device="cpu")
    suffix = ".forward_linear.qweight"
    names = sorted(key.removesuffix(suffix) for key in state if key.endswith(suffix))
    if not names:
        raise ValueError("dual checkpoint contains no forward_linear weights")
    for name in names:
        target = model.get_submodule(name)
        base = target.get_base_layer() if hasattr(target, "get_base_layer") else target
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is not a Linear base layer")
        forward_linear = _make_nunchaku_linear(
            state, f"{name}.forward_linear", base.in_features, base.out_features
        )
        backward_linear = _make_nunchaku_linear(
            state, f"{name}.backward_linear", base.out_features, base.in_features
        )
        _set_base_layer(model, name, DualSVDQuantLinear(forward_linear, backward_linear))
    return len(names)
