"""FP8 Linear layer with pre-quantized weights for LoRA training.

The base (frozen) weight is stored in FP8 with a per-tensor scale, computed
once at checkpoint creation time. During training:

  Forward:  activation → dynamic-scale FP8 cast → torch._scaled_mm(fp8_act, fp8_weight)
  Backward: grad_output → dynamic-scale FP8 cast → torch._scaled_mm(fp8_grad, fp8_weight)

Weight quantization cost: ZERO (pre-computed at load time from the FP8 checkpoint).
Activation quantization cost: one amax reduction + one elementwise cast per layer
(small — activations are ~10x smaller than weights for this model's shapes).

This module is an nn.Linear subclass so ai-toolkit's LoRA injection still finds
and wraps it. The LoRA delta (computed in BF16) is added to the FP8 base output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


FP8_MAX = 448.0


def _quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor FP8 quantization. Returns (fp8_tensor, scale_scalar).

    The scale is a plain float32 tensor (not 0-dim) to satisfy torch._scaled_mm's
    tensorwise-scaling requirement: scales must be singletons but 1-element tensors.
    """
    amax = x.abs().amax()
    scale = (amax / FP8_MAX).clamp(min=1e-12).float()
    # Cast in the input dtype (bf16) — do NOT upcast the whole activation to fp32 first.
    # The fp32 intermediate was ~doubling the memory traffic of the per-step quant, which
    # is the dominant per-step fp8 overhead on this bandwidth-limited box.
    inv = (1.0 / scale).to(x.dtype)
    fp8 = (x * inv).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return fp8, scale


class FP8Linear(nn.Linear):
    """nn.Linear with a cached FP8 weight for accelerated GEMM.

    The ``weight`` parameter is normally emptied after FP8 caching to save VRAM
    (weights are frozen for LoRA training). Set ``use_fp8=False`` to fall back to
    standard F.linear (requires weight to be materialized first).
    """

    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self._fp8_weight: torch.Tensor = None  # (out, in) in float8_e4m3fn
        self._fp8_weight_t: torch.Tensor = None  # (in, out) in float8_e4m3fn
        self._fp8_scale: torch.Tensor = None   # (1,) float32
        self.use_fp8 = True

    def cache_fp8(self):
        """Quantize self.weight to FP8 and cache. Call once after weight is loaded."""
        w = self.weight.detach()
        amax = w.abs().amax()
        self._fp8_scale = (amax / FP8_MAX).clamp(min=1e-12).float()
        fp8 = (w.float() / self._fp8_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        self._fp8_weight = fp8.contiguous()
        self._fp8_weight_t = fp8.t().contiguous()

    def load_fp8(self, fp8_weight: torch.Tensor, fp8_scale: torch.Tensor,
                 drop_bf16: bool = True, store_transposed: bool = False):
        """Load pre-quantized FP8 weight + scale (from checkpoint).

        Args:
            drop_bf16: Free the BF16 self.weight to save VRAM.
            store_transposed: Also store (K,N) contiguous copy for FP8 backward.
                When False (default), backward dequantizes to BF16 on-the-fly —
                saves ~12GB VRAM at the cost of a BF16 backward GEMM.
        """
        self._fp8_scale = fp8_scale.to(self.weight.device).float()
        fp8 = fp8_weight.to(self.weight.device)
        self._fp8_weight = fp8.contiguous()       # (N,K) for forward
        if store_transposed:
            self._fp8_weight_t = fp8.t().contiguous()  # (K,N) for FP8 backward
        else:
            self._fp8_weight_t = None  # backward will use BF16 dequant path
        if drop_bf16:
            self.weight = nn.Parameter(torch.empty(0, device=self.weight.device,
                                                   dtype=self.weight.dtype),
                                       requires_grad=False)

    def ensure_bf16_weight(self):
        """Lazily reconstruct BF16 weight from FP8 cache (for save/merge)."""
        if self.weight.numel() == 0 and self._fp8_weight is not None:
            self.weight = nn.Parameter(
                (self._fp8_weight.float() * self._fp8_scale).to(self._fp8_weight.device).bfloat16(),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_fp8 or self._fp8_weight is None or not x.is_cuda:
            return super().forward(x)

        return _FP8LinearFn.apply(
            x, self._fp8_weight, self._fp8_weight_t, self._fp8_scale, self.bias
        )


class _FP8LinearFn(torch.autograd.Function):
    """FP8 matmul with dynamic activation scaling, pre-cached FP8 weight.

    Forward:  out = (act_fp8 @ w_fp8^T) * act_scale * w_scale  [→ bf16]
    Backward: grad_x = (grad_out_fp8 @ w_fp8) * grad_scale * w_scale  [→ bf16]
              grad_w = None (weight is frozen for LoRA training)
    """

    @staticmethod
    def forward(ctx, x, fp8_weight, fp8_weight_t, fp8_scale, bias):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])

        x_fp8, x_scale = _quantize_fp8(x2d)

        out = torch._scaled_mm(
            x_fp8, fp8_weight.t(),
            scale_a=x_scale, scale_b=fp8_scale,
            out_dtype=torch.bfloat16,
        )

        if bias is not None:
            out = out + bias

        ctx.save_for_backward(fp8_weight, fp8_weight_t, fp8_scale)
        return out.reshape(*orig_shape[:-1], fp8_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        fp8_weight, fp8_weight_t, fp8_scale = ctx.saved_tensors

        grad_shape = grad_out.shape
        grad2d = grad_out.reshape(-1, grad_shape[-1])

        if fp8_weight_t is not None:
            # FP8 backward path (uses stored transposed weight)
            g_fp8, g_scale = _quantize_fp8(grad2d)
            grad_x = torch._scaled_mm(
                g_fp8, fp8_weight_t.t(),
                scale_a=g_scale, scale_b=fp8_scale,
                out_dtype=torch.bfloat16,
            )
        else:
            # BF16 backward path (dequantize on-the-fly, saves 12GB VRAM)
            w_bf16 = (fp8_weight.float() * fp8_scale).bfloat16()
            grad_x = torch.mm(grad2d, w_bf16)

        grad_bias = grad_out.reshape(-1, grad_out.shape[-1]).sum(dim=0) if ctx.needs_input_grad[4] else None
        return grad_x.reshape(*grad_shape[:-1], fp8_weight.shape[1]), None, None, None, grad_bias
