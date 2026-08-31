# The krea2_edit geometry contract

This document pins the *exact* reference geometry and conditioning semantics of
`krea2_edit` training, so that inference stacks and third-party trainers can verify
parity instead of guessing. The [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit)
nodes (v1.2.4+) implement the same contract; LoRAs trained against a different
geometry will misregister at inference (blur, off-center ghosting, seam artifacts).

## 1. Token sequence

```
[ text tokens | ref_1 tokens | ... | ref_N tokens | target tokens ]
```

- `N ≤ 2`: the inference nodes expose exactly two reference inputs (`source_image`,
  `source_image_b`), so the trainer refuses more than two references.
- The transformer input is one joint sequence. References are clean (un-noised) VAE
  latents; only target tokens receive flow-matching noise, and **only target tokens
  contribute to the loss** (the predicted-flow slice for refs is discarded).
- Latents are patchified 2×2 on a /8 VAE, so one token = 16×16 pixels.

## 2. 3D RoPE position ids

Each image token carries `(frame, h, w)`:

- **Target: frame 0.** References: **frame i+1** in list order (ref_1 → frame 1, …).
- Text tokens: all-zero position ids.
- Target grid: `h ∈ [0, H/16)`, `w ∈ [0, W/16)`, stride 1.

## 3. Independent native reference grids

References are neither cropped nor resized to the target. For a reference token grid
`Hr × Wr`, its positions are exactly:

```
(frame=i+1, h=0..Hr-1, w=0..Wr-1)
```

Target positions remain `(frame=0, h=0..Ht-1, w=0..Wt-1)`. Thus each image has an
independent 2D coordinate system and frame index; no target-relative scale or centered
offset is encoded. At pixel ingest, only bottom/right replicate padding to a multiple
of 16 pixels is applied so the /8 VAE and 2×2 latent DiT patching divide exactly. No
source pixel is removed or resampled.

## 4. Grounded text conditioning

The instruction is encoded by Qwen3-VL-4B **together with the reference images**:

- Template: fixed system prefix (describe color/shape/size/texture/…), then the user
  turn containing one `<|vision_start|><|image_pad|><|vision_end|>` block **per
  reference, in frame order**, followed by the instruction text.
- Output: hidden states of 12 selected layers `(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)`
  stacked to `(1, L, 12, hidden)`, with the first 34 tokens (system prefix) sliced off.
- The grounding image is the reference as delivered by the loader, at its **native
  resolution**. The appearance path uses the same content, with only 16-pixel edge
  padding before VAE encoding.
- The grounding image is then optionally downscaled: longest side capped at
  `GROUNDING_MAX_PX` (default **768**, matching the inference node's `grounding_px`)
  with per-step uniform jitter down to `GROUNDING_JITTER_MIN` (default **384**).
  Both env vars override the defaults; `0` disables the cap. The downscale kernel is
  PIL `LANCZOS` here; the inference node uses `common_upscale(..., "area")`. Jitter is
  a train-time augmentation — which is why
  **text-embedding caching must stay off** for edit datasets: a cached embedding
  freezes one grounding scale and the LoRA loses scale robustness.
- **Cache invalidation is manual.** ai-toolkit's text-embedding cache key does not
  include the grounding settings, so changing `GROUNDING_MAX_PX` /
  `GROUNDING_JITTER_MIN` does *not* invalidate an existing cache. If you train with
  `cache_text_embeddings: true` and change either value, delete the `_t_e_cache`
  folders under your dataset directories first — otherwise the run keeps feeding
  embeddings built at the previous grounding resolution and this section no longer
  describes what the model actually sees.

## 5. Objective

- Flow matching, velocity target `v = noise − clean`, MSE on target tokens only.
- `timestep_type: "weighted"` = **uniform** timestep sampling + a per-timestep loss
  weight lookup. It is not a sampling distribution (and not sigmoid/logit-normal).

## 6. Checkpoint format

LoRA keys follow the ComfyUI convention (`diffusion_model.<module>.lora_…` + per-module
`alpha`), targeting the transformer's attention/MLP linears including the text-fusion
blocks. Checkpoints load directly in ComfyUI via the standard LoRA loader.
