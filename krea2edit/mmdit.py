"""Krea 2 (K2) single-stream MMDiT backbone.

Vendored from the Krea 2 reference ``mmdit.py``. This is a single-stream
MMDiT: Qwen3-VL text features are fused by a small ``TextFusionTransformer`` and
then concatenated with the patchified image latent tokens into one sequence that
flows through ``SingleStreamBlock`` layers. The model predicts the flow-matching
velocity on the image tokens.

Differences from the reference (all training-driven, numerically equivalent):
  - ``torch.compile`` decorators are dropped (they fight gradient checkpointing,
    LoRA module swapping and variable shapes during training).
  - Regular attention uses ``F.scaled_dot_product_attention`` instead of forcing
    the cuDNN SDPA backend, so it works across dtypes / masks / backward. Ragged
    B>1 edit batches use FlashAttention-2/4 packed varlen attention instead.
  - ``enable_gradient_checkpointing`` / ``disable_gradient_checkpointing`` and a
    per-block ``torch.utils.checkpoint`` wrapper are added (gated on
    ``torch.is_grad_enabled()`` so eval/sampling never pays for it).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def ropeapply_packed(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    """RoPE for a varlen-packed token stream.

    ``flash_attn_varlen_func`` receives ``(total_tokens, heads, head_dim)``
    instead of the regular ``(batch, heads, seq, head_dim)`` layout.  The RoPE
    frequencies are packed in exactly the same token order.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def _flash_attention4_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> Tensor:
    """Invoke the FA4 CUTE varlen API (Hopper/Blackwell)."""
    from flash_attn.cute import flash_attn_varlen_func
    return flash_attn_varlen_func(
        q.contiguous(), k.contiguous(), v.contiguous(),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
    )


def _flash_attention2_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> Tensor:
    """Invoke the FlashAttention-2 varlen API (Ampere/Ada/Hopper)."""
    from flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func(
        q.contiguous(), k.contiguous(), v.contiguous(),
        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        dropout_p=0.0, causal=False,
    )


def flash_attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> Tensor:
    """Select FA2/FA4 varlen attention from the CUDA architecture.

    A100/A800 (SM80) use FA2. Hopper and newer (SM90+) prefer FA4; if FA4 is
    absent, FA2 is tried as a compatible fallback where the installed build supports
    that GPU. There is deliberately no padded-SDPA fallback for ragged B>1 inputs.
    """
    if not q.is_cuda:
        raise RuntimeError("Krea2Edit ragged training requires CUDA FlashAttention.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "Krea2Edit ragged training requires fp16 or bf16 Q/K/V for FlashAttention; "
            f"got {q.dtype}."
        )
    major, minor = torch.cuda.get_device_capability(q.device)
    arch = f"sm_{major}{minor}"
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()

    if major >= 9:
        try:
            return _flash_attention4_varlen(q, k, v, cu_seqlens, max_seqlen)
        except ImportError:
            # FA2 remains useful on Hopper when a FA4 wheel has not yet been built
            # for a user's CUDA stack. It is not a padding fallback: it stays varlen.
            try:
                return _flash_attention2_varlen(q, k, v, cu_seqlens, max_seqlen)
            except ImportError as exc:
                raise RuntimeError(
                    f"Krea2Edit ragged training on {arch} needs FlashAttention-4 "
                    "(`pip install flash-attn-4`) or a compatible FlashAttention-2 build "
                    "(`pip install flash-attn`)."
                ) from exc

    if major == 8:
        try:
            return _flash_attention2_varlen(q, k, v, cu_seqlens, max_seqlen)
        except ImportError as exc:
            raise RuntimeError(
                f"Krea2Edit ragged training on {arch} (including A100/A800) needs "
                "FlashAttention-2: `pip install flash-attn`."
            ) from exc

    raise RuntimeError(
        f"Krea2Edit ragged training needs FlashAttention varlen, but {arch} is unsupported. "
        "Use an Ampere (A100/A800), Ada, Hopper, or newer NVIDIA GPU, or batch_size: 1."
    )


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
) -> Tensor:
    # Let PyTorch auto-select the best SDPA backend. Forcing CUDNN_ATTENTION is
    # catastrophically slow on Blackwell (GB10 / RTX 5090) and often slower than
    # flash/memory-efficient on other architectures for diffusion attention shapes.
    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
    )
    return rearrange(x, "B H L D -> B L (H D)")


def _mask(mask: Tensor) -> Tensor:
    """Expand a (B, L) key-padding mask into a (B, 1, L, L) attention mask."""
    return mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(period)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    # t: (B,) -> args: (B, 1, half), so the embedding broadcasts as a per-sample vec.
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20


class SimpleModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    # vec (b d)
    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))

    # vec (b (6 d))
    def forward(self, vec: Tensor):
        out = vec + self.lin
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(
            6, dim=-1
        )
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims  # how to split the head dimension across the position axes
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [
                rope(pos[..., i], d, self.theta, self.ntk)
                for i, d in enumerate(self.axdims)
            ],
            dim=-3,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class RMSNorm(torch.nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.zeros(features, device=device, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(
            t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0)
        )
        return t.to(dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self, features: int, multiplier: int, bias: bool = False, multiple: int = 128
    ):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)

        self.gate = torch.nn.Linear(features, mlpdim, bias=bias)
        self.up = torch.nn.Linear(features, mlpdim, bias=bias)
        self.down = torch.nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(torch.nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads

        self.wq = torch.nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = torch.nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = torch.nn.Linear(dim, dim, bias=bias)

    def forward(
        self, qkv: Tensor, freqs: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)

        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        out = self.wo(attention(q, k, v, mask=mask, gqa=self.gqa) * F.sigmoid(gate))

        return out

    def forward_varlen(
        self,
        qkv: Tensor,
        freqs: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Tensor:
        """Packed FA2/FA4 attention for ragged batches (no pad tokens, no mask)."""
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
        q = rearrange(q, "T (H D) -> T H D", H=self.heads)
        k = rearrange(k, "T (H D) -> T H D", H=self.kvheads)
        v = rearrange(v, "T (H D) -> T H D", H=self.kvheads)
        q, k, v = self.qknorm(q, k, v)
        q, k = ropeapply_packed(q, k, freqs)
        out = flash_attention_varlen(q, k, v, cu_seqlens, max_seqlen)
        out = rearrange(out, "T H D -> T (H D)")
        return self.wo(out * F.sigmoid(gate))


class LastLayer(torch.nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = torch.nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(
        self,
        x: Tensor,
        tvec: Tensor,
        alternate_tvec: Tensor | None = None,
        alternate_mask: Tensor | None = None,
    ) -> Tensor:
        scale, shift = self.modulation(tvec)
        normalized = self.norm(x)
        x = (1 + scale) * normalized + shift
        if alternate_tvec is not None:
            if alternate_mask is None or alternate_mask.shape != x.shape[:2]:
                raise ValueError(
                    "alternate final-layer mask must have shape (B, target_tokens)"
                )
            alternate_scale, alternate_shift = self.modulation(alternate_tvec)
            batch_ids, token_ids = torch.where(alternate_mask)
            x[batch_ids, token_ids] = (
                (1 + alternate_scale[batch_ids, 0])
                * normalized[batch_ids, token_ids]
                + alternate_shift[batch_ids, 0]
            )
        x = self.linear(x)
        return x

    def forward_varlen(
        self,
        x: Tensor,
        tvec: Tensor,
        sample_ids: Tensor,
        alternate_tvec: Tensor | None = None,
        alternate_mask: Tensor | None = None,
    ) -> Tensor:
        """Final projection for selected packed tokens.

        ``tvec`` remains one vector per sample; only the target tokens are sent
        here, indexed by their originating sample.
        """
        scale, shift = self.modulation(tvec)
        scale, shift = scale[:, 0, :], shift[:, 0, :]
        normalized = self.norm(x)
        x = (1 + scale[sample_ids]) * normalized + shift[sample_ids]
        if alternate_tvec is not None:
            if alternate_mask is None or alternate_mask.shape != x.shape[:1]:
                raise ValueError(
                    "packed alternate final-layer mask must match target tokens"
                )
            alternate_scale, alternate_shift = self.modulation(alternate_tvec)
            alternate_scale = alternate_scale[:, 0, :][sample_ids]
            alternate_shift = alternate_shift[:, 0, :][sample_ids]
            x = torch.where(
                alternate_mask[:, None],
                (1 + alternate_scale) * normalized + alternate_shift,
                x,
            )
        return self.linear(x)


class TextFusionBlock(torch.nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))

        return x


class TextFusionTransformer(torch.nn.Module):
    # num_txt_layers is the number of selected encoder hidden-state layers fed in
    # (projected down to 1), NOT the transformer depth — that's fixed at 2 + 2 blocks.
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        # Collapse to 3D for the projector: a quantized (quanto) Linear's matmul
        # kernel only accepts 2D/3D activations, and this layer-axis projection
        # (n -> 1) otherwise feeds it a 4D (b, l, d, n) tensor.
        x = self.projector(x.reshape(b * l, d, n))
        x = x.reshape(b, l, d)

        for block in self.refiner_blocks:
            x = block(x, mask=mask)

        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self,
        x: Tensor,
        normal_vec: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
        clean_vec: Tensor | None = None,
        clean_token_range: tuple[int, int] | None = None,
        alternate_vec: Tensor | None = None,
        alternate_token_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply normal, alternate target, and clean-reference modulation."""
        normal = self.mod(normal_vec)
        alternate = self.mod(alternate_vec) if alternate_vec is not None else None
        clean = self.mod(clean_vec) if clean_vec is not None else None

        if alternate is not None and (
            alternate_token_mask is None
            or alternate_token_mask.shape != x.shape[:2]
        ):
            raise ValueError("alternate token mask must have shape (B, sequence)")
        if clean_token_range is not None:
            if clean is None:
                raise ValueError("clean_token_range requires clean modulation vectors")
            start, end = clean_token_range
            if not (0 <= start < end <= x.shape[1]):
                raise ValueError(
                    f"invalid clean token range [{start}, {end}) for sequence length {x.shape[1]}"
                )

        def affine(
            normalized: Tensor,
            normal_scale: Tensor,
            normal_shift: Tensor,
            alternate_scale: Tensor | None,
            alternate_shift: Tensor | None,
            clean_scale: Tensor | None,
            clean_shift: Tensor | None,
        ) -> Tensor:
            result = (1 + normal_scale) * normalized + normal_shift
            if alternate_scale is not None:
                batch_ids, token_ids = torch.where(alternate_token_mask)
                result[batch_ids, token_ids] = (
                    (1 + alternate_scale[batch_ids, 0])
                    * normalized[batch_ids, token_ids]
                    + alternate_shift[batch_ids, 0]
                )
            if clean_scale is not None:
                start, end = clean_token_range
                result[:, start:end] = (
                    (1 + clean_scale) * normalized[:, start:end] + clean_shift
                )
            return result

        def gate(
            update: Tensor,
            normal_gate: Tensor,
            alternate_gate: Tensor | None,
            clean_gate: Tensor | None,
        ) -> Tensor:
            result = normal_gate * update
            if alternate_gate is not None:
                batch_ids, token_ids = torch.where(alternate_token_mask)
                result[batch_ids, token_ids] = (
                    alternate_gate[batch_ids, 0] * update[batch_ids, token_ids]
                )
            if clean_gate is not None:
                start, end = clean_token_range
                result[:, start:end] = clean_gate * update[:, start:end]
            return result

        alternate_values = alternate or (None,) * 6
        clean_values = clean or (None,) * 6
        pre_norm = self.prenorm(x)
        pre = affine(
            pre_norm,
            normal[0],
            normal[1],
            alternate_values[0],
            alternate_values[1],
            clean_values[0],
            clean_values[1],
        )
        x = x + gate(
            self.attn(pre, freqs, mask),
            normal[2],
            alternate_values[2],
            clean_values[2],
        )
        post_norm = self.postnorm(x)
        post = affine(
            post_norm,
            normal[3],
            normal[4],
            alternate_values[3],
            alternate_values[4],
            clean_values[3],
            clean_values[4],
        )
        return x + gate(
            self.mlp(post),
            normal[5],
            alternate_values[5],
            clean_values[5],
        )

    def forward_varlen(
        self,
        x: Tensor,
        normal_vec: Tensor,
        clean_vec: Tensor,
        freqs: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        sample_ids: Tensor,
        clean_mask: Tensor,
        alternate_vec: Tensor | None = None,
        alternate_mask: Tensor | None = None,
    ) -> Tensor:
        """AdaLN-single block on a packed batch of independent sequences.

        The reference-token t=0 treatment is expressed as a boolean packed mask,
        rather than a common token span.  This matters when every sample has a
        different text length, number of references, or reference resolution.
        """
        normal = self.mod(normal_vec)
        clean = self.mod(clean_vec)
        alternate = self.mod(alternate_vec) if alternate_vec is not None else None
        if alternate is not None and (
            alternate_mask is None or alternate_mask.shape != clean_mask.shape
        ):
            raise ValueError("packed alternate mask must match the packed sequence")

        def choose(
            normal_value: Tensor,
            clean_value: Tensor,
            alternate_value: Tensor | None,
        ) -> Tensor:
            normal_value = normal_value[:, 0, :][sample_ids]
            clean_value = clean_value[:, 0, :][sample_ids]
            if alternate_value is not None:
                alternate_value = alternate_value[:, 0, :][sample_ids]
                normal_value = torch.where(
                    alternate_mask[:, None], alternate_value, normal_value
                )
            return torch.where(clean_mask[:, None], clean_value, normal_value)

        alternate_values = alternate or (None,) * 6
        prescale, preshift, pregate, postscale, postshift, postgate = (
            choose(n, c, a)
            for n, c, a in zip(normal, clean, alternate_values)
        )
        pre = (1 + prescale) * self.prenorm(x) + preshift
        x = x + pregate * self.attn.forward_varlen(
            pre, freqs, cu_seqlens, max_seqlen
        )
        post = (1 + postscale) * self.postnorm(x) + postshift
        return x + postgate * self.mlp(post)


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        headdim = config.features // config.heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"

        self.posemb = PositionalEncoding(
            config.features, axes, theta=config.theta, ntk=1.0
        )
        self.first = nn.Linear(
            config.channels * config.patch**2, config.features, bias=True
        )

        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features,
                    config.heads,
                    config.multiplier,
                    config.bias,
                    config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)

        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        ref_token_count: int = 0,
        alternate_t: Tensor | None = None,
        alternate_target_mask: Tensor | None = None,
        representation_layer: int | None = None,
        return_velocity: bool = True,
    ) -> Tensor | tuple[Tensor | None, Tensor]:
        if representation_layer is not None and not (
            1 <= representation_layer <= len(self.blocks)
        ):
            raise ValueError(
                f"representation_layer must be in [1, {len(self.blocks)}]"
            )
        if not return_velocity and representation_layer is None:
            raise ValueError("return_velocity=False requires representation_layer")
        img = self.first(img)
        flow_t = t
        normal_t = self.tmlp(
            temb(flow_t, self.config.tdim, device=img.device, dtype=img.dtype)
        )
        normal_vec = self.tproj(normal_t)
        alternate_t_embedding = None
        alternate_vec = None
        if alternate_t is not None:
            if alternate_t.shape != flow_t.shape:
                raise ValueError("alternate_t must have the same shape as t")
            alternate_t_embedding = self.tmlp(
                temb(
                    alternate_t,
                    self.config.tdim,
                    device=img.device,
                    dtype=img.dtype,
                )
            )
            alternate_vec = self.tproj(alternate_t_embedding)

        _tm = mask[:, : context.shape[1]]
        txtmask = None if bool(_tm.all()) else _mask(_tm)

        context = self.txtfusion(context, mask=txtmask)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        targetlen = imglen - ref_token_count
        if targetlen <= 0:
            raise ValueError("the image stream must contain at least one target token")
        combined = torch.cat((context, img), dim=1)
        combined_alternate_mask = None
        if alternate_vec is not None:
            if (
                alternate_target_mask is None
                or alternate_target_mask.shape != (img.shape[0], targetlen)
            ):
                raise ValueError(
                    "alternate_target_mask must have shape (B, target_tokens)"
                )
            combined_alternate_mask = torch.cat(
                (
                    torch.zeros(
                        (img.shape[0], txtlen),
                        dtype=torch.bool,
                        device=img.device,
                    ),
                    alternate_target_mask,
                    torch.zeros(
                        (img.shape[0], ref_token_count),
                        dtype=torch.bool,
                        device=img.device,
                    ),
                ),
                dim=1,
            )
        clean_token_range = None
        clean_vec = None
        if ref_token_count:
            if not 0 < ref_token_count < imglen:
                raise ValueError(
                    f"ref_token_count must be in [1, {imglen - 1}], got {ref_token_count}"
                )
            # The image sequence is [noisy target tokens | reference tokens]. Text
            # and target retain the sampled flow time; the reference suffix uses t=0.
            clean_t = self.tmlp(
                temb(
                    torch.zeros_like(flow_t),
                    self.config.tdim,
                    device=img.device,
                    dtype=img.dtype,
                )
            )
            clean_vec = self.tproj(clean_t)
            clean_token_range = (
                txtlen + imglen - ref_token_count,
                txtlen + imglen,
            )

        # Pad combined sequence to a multiple of 256 to stabilize compiled kernel shapes.
        # Off by default: the trailing pad positions must be masked, and a dense pad-mask
        # forces SDPA off its fused/flash path. Re-enable (with a fused-compatible mask)
        # only under torch.compile, where static shapes matter.
        fulllen = combined.shape[1]
        _padlen = (-fulllen) % 256 if getattr(self, "pad_seq_to_256", False) else 0
        if _padlen > 0:
            combined = F.pad(combined, (0, 0, 0, _padlen))
            mask = F.pad(mask, (0, _padlen), value=False)
            pos = F.pad(pos, (0, 0, 0, _padlen))
            if combined_alternate_mask is not None:
                combined_alternate_mask = F.pad(
                    combined_alternate_mask, (0, _padlen), value=False
                )

        # A dense (B,1,L,L) attn_mask forces SDPA into an O(L^2)-memory kernel (SMs ~100%
        # busy but tensor cores idle / low power). When every token is valid (batch=1, no
        # padding) the mask is a no-op -> pass None so cuDNN/flash fuses. Materialize the
        # dense mask only when there is real padding to honor.
        mask = None if bool(mask.all()) else _mask(mask)

        freqs = self.posemb(pos)

        representation = None
        for i, block in enumerate(self.blocks):
            do_ckpt = (
                self.gradient_checkpointing
                and torch.is_grad_enabled()
                and (i % max(1, getattr(self, "checkpoint_every_n", 1)) == 0)
            )
            if do_ckpt:
                combined = checkpoint(
                    block,
                    combined,
                    normal_vec,
                    freqs,
                    mask,
                    clean_vec,
                    clean_token_range,
                    alternate_vec,
                    combined_alternate_mask,
                    use_reentrant=False,
                )
            else:
                combined = block(
                    combined,
                    normal_vec,
                    freqs,
                    mask,
                    clean_vec,
                    clean_token_range,
                    alternate_vec,
                    combined_alternate_mask,
                )
            if representation_layer == i + 1:
                representation = combined[:, txtlen : txtlen + targetlen]
                if not return_velocity:
                    return None, representation

        target_hidden = combined[:, txtlen : txtlen + targetlen]
        output = self.last(
            target_hidden,
            normal_t,
            alternate_t_embedding,
            alternate_target_mask,
        )

        if representation_layer is not None:
            return output, representation
        return output

    def forward_varlen(
        self,
        image_tokens: list[Tensor],
        context: list[Tensor],
        t: Tensor,
        image_positions: list[Tensor],
        ref_token_counts: list[int],
        alternate_t: Tensor | None = None,
        alternate_target_masks: list[Tensor] | None = None,
        representation_layer: int | None = None,
        return_velocity: bool = True,
    ) -> list[Tensor] | tuple[list[Tensor] | None, list[Tensor]]:
        """Run the DiT over a ragged batch with FlashAttention-2/4 varlen kernels.

        Each element describes exactly one independent sequence:
        ``[text_i | target_i | ref_1_i | ... | ref_N_i]``.  Only the main
        image/text DiT stream is packed; the tiny four-block text-fusion module is
        evaluated per sample so it never needs padding either.  Returned tensors are
        target-only patch predictions, one ``(L_target_i, patch_out)`` tensor each.
        """
        batch_size = len(image_tokens)
        if not (
            batch_size == len(context) == len(image_positions) == len(ref_token_counts)
        ):
            raise ValueError("varlen inputs must contain one image/context/position entry per sample")
        if batch_size == 0:
            return []
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(
                f"varlen timestep must be (B,), got {tuple(t.shape)} for B={batch_size}"
            )
        if representation_layer is not None and not (
            1 <= representation_layer <= len(self.blocks)
        ):
            raise ValueError(
                f"representation_layer must be in [1, {len(self.blocks)}]"
            )
        if not return_velocity and representation_layer is None:
            raise ValueError("return_velocity=False requires representation_layer")
        if alternate_t is not None:
            if alternate_t.shape != t.shape:
                raise ValueError("alternate_t must have the same shape as t")
            if (
                alternate_target_masks is None
                or len(alternate_target_masks) != batch_size
            ):
                raise ValueError(
                    "alternate_target_masks must contain one mask per sample"
                )

        sequences: list[Tensor] = []
        positions: list[Tensor] = []
        clean_masks: list[Tensor] = []
        alternate_masks: list[Tensor] = []
        target_masks: list[Tensor] = []
        sample_ids: list[Tensor] = []
        lengths: list[int] = []

        for sample, (img, ctx, img_pos, ref_len) in enumerate(
            zip(image_tokens, context, image_positions, ref_token_counts)
        ):
            if img.ndim != 2 or img_pos.ndim != 2 or img_pos.shape[-1] != 3:
                raise ValueError("varlen image tokens must be (L,C) with positions (L,3)")
            if img.shape[0] != img_pos.shape[0]:
                raise ValueError("varlen image token and position counts differ")
            if not 0 <= ref_len < img.shape[0]:
                raise ValueError(
                    f"sample {sample}: ref_token_count must be in [0, {img.shape[0] - 1}], got {ref_len}"
                )
            if ctx.ndim != 3:
                raise ValueError("varlen text context must be (L_text, selected_layers, hidden)")

            # Keeping B=1 here avoids padding variable VLM-language contexts. It is
            # only four small text blocks; the 48 DiT blocks below use packed FA2/FA4.
            txt = self.txtmlp(self.txtfusion(ctx.unsqueeze(0), mask=None).squeeze(0))
            img = self.first(img)
            txt_len, img_len = txt.shape[0], img.shape[0]
            target_len = img_len - ref_len
            seq_len = txt_len + img_len

            sequences.append(torch.cat((txt, img), dim=0))
            positions.append(torch.cat((
                torch.zeros((txt_len, 3), dtype=img_pos.dtype, device=img_pos.device),
                img_pos,
            ), dim=0))
            clean_masks.append(torch.cat((
                torch.zeros(txt_len + target_len, dtype=torch.bool, device=img.device),
                torch.ones(ref_len, dtype=torch.bool, device=img.device),
            )))
            if alternate_t is not None:
                alternate_target_mask = alternate_target_masks[sample]
                if alternate_target_mask.shape != (target_len,):
                    raise ValueError(
                        f"sample {sample}: alternate target mask must have "
                        f"shape ({target_len},)"
                    )
                alternate_masks.append(torch.cat((
                    torch.zeros(txt_len, dtype=torch.bool, device=img.device),
                    alternate_target_mask.to(device=img.device, dtype=torch.bool),
                    torch.zeros(ref_len, dtype=torch.bool, device=img.device),
                )))
            target_masks.append(torch.cat((
                torch.zeros(txt_len, dtype=torch.bool, device=img.device),
                torch.ones(target_len, dtype=torch.bool, device=img.device),
                torch.zeros(ref_len, dtype=torch.bool, device=img.device),
            )))
            sample_ids.append(torch.full((seq_len,), sample, dtype=torch.long, device=img.device))
            lengths.append(seq_len)

        packed = torch.cat(sequences, dim=0)
        packed_pos = torch.cat(positions, dim=0)
        packed_clean_mask = torch.cat(clean_masks, dim=0)
        packed_alternate_mask = (
            torch.cat(alternate_masks, dim=0) if alternate_masks else None
        )
        packed_target_mask = torch.cat(target_masks, dim=0)
        packed_sample_ids = torch.cat(sample_ids, dim=0)
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        cu_seqlens = torch.tensor(cumulative, dtype=torch.int32, device=packed.device)
        max_seqlen = max(lengths)

        normal_t = self.tmlp(
            temb(t, self.config.tdim, device=packed.device, dtype=packed.dtype)
        )
        normal_vec = self.tproj(normal_t)
        alternate_t_embedding = None
        alternate_vec = None
        if alternate_t is not None:
            alternate_t_embedding = self.tmlp(
                temb(
                    alternate_t,
                    self.config.tdim,
                    device=packed.device,
                    dtype=packed.dtype,
                )
            )
            alternate_vec = self.tproj(alternate_t_embedding)
        clean_t = self.tmlp(
            temb(torch.zeros_like(t), self.config.tdim, device=packed.device, dtype=packed.dtype)
        )
        clean_vec = self.tproj(clean_t)
        freqs = self.posemb(packed_pos.unsqueeze(0)).squeeze(0)

        representation_tokens = None
        for i, block in enumerate(self.blocks):
            do_ckpt = (
                self.gradient_checkpointing
                and torch.is_grad_enabled()
                and (i % max(1, getattr(self, "checkpoint_every_n", 1)) == 0)
            )
            if do_ckpt:
                def block_forward(tokens, block=block):
                    return block.forward_varlen(
                        tokens, normal_vec, clean_vec, freqs, cu_seqlens, max_seqlen,
                        packed_sample_ids, packed_clean_mask, alternate_vec,
                        packed_alternate_mask,
                    )
                packed = checkpoint(block_forward, packed, use_reentrant=False)
            else:
                packed = block.forward_varlen(
                    packed, normal_vec, clean_vec, freqs, cu_seqlens, max_seqlen,
                    packed_sample_ids, packed_clean_mask, alternate_vec,
                    packed_alternate_mask,
                )
            if representation_layer == i + 1:
                representation_tokens = packed[packed_target_mask]
                if not return_velocity:
                    target_sample_ids = packed_sample_ids[packed_target_mask]
                    representations = [
                        representation_tokens[target_sample_ids == sample]
                        for sample in range(batch_size)
                    ]
                    return None, representations

        target_sample_ids = packed_sample_ids[packed_target_mask]
        target_alternate_mask = (
            packed_alternate_mask[packed_target_mask]
            if packed_alternate_mask is not None
            else None
        )
        target_tokens = self.last.forward_varlen(
            packed[packed_target_mask],
            normal_t,
            target_sample_ids,
            alternate_t_embedding,
            target_alternate_mask,
        )
        outputs = [
            target_tokens[target_sample_ids == sample]
            for sample in range(batch_size)
        ]
        if representation_layer is not None:
            representations = [
                representation_tokens[target_sample_ids == sample]
                for sample in range(batch_size)
            ]
            return outputs, representations
        return outputs
