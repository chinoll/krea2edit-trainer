# krea2edit trainer — ai-toolkit extension

*Community project — not affiliated with or endorsed by Krea.ai. "Krea" is used
descriptively to identify the base model this trainer targets.*

☕ **[Support on Ko-fi](https://ko-fi.com/conradlocke)** — all tips go straight to GPU
compute for future versions.

This fork uses an **independent native-grid reference geometry**: every reference
keeps its own aspect ratio and starts its own `(h=0, w=0)` RoPE grid. References are
never cropped or resized to the target; only bottom/right edge padding aligns them to
the 16-pixel VAE/DiT lattice. The paired
[comfyui-krea2edit](https://github.com/chinoll/comfyui-krea2edit) fork implements the
same contract, so what you train is what the nodes run.

There is no fixed DiT spatial-position table: targets use their own H×W grid from the
ai-toolkit dataset resolution/buckets, while every source retains an independent native
grid. The paired nodes can sample any requested output pixel size by alignment-padding
internally and cropping that padding after VAE decode; VRAM and the trained resolution
range, rather than a hard geometry limit, are the practical constraints.

This is intentionally incompatible with the released
[krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit) LoRAs,
which used target-fitted/centered reference geometry. Train a new LoRA for this mode.

It adds one model architecture to [ai-toolkit](https://github.com/ostris/ai-toolkit):

- `krea2_edit` — instruction-based, reference-conditioned editing (the identity-edit
  recipe): in-context reference tokens + image-grounded text conditioning

(Plain Krea 2 text-to-image training is already built into upstream ai-toolkit as
`arch: "krea2"` — this extension doesn't duplicate it.)

**Note on upstream's own edit mode:** ai-toolkit's built-in `krea2` arch also offers
an edit mode (`model_kwargs: {edit: true}`). It is a *different training contract* —
a "Picture N:"-labeled grounding template and an area-budget reference resize —
whereas this extension implements the exact grounding template and independent
native-grid reference geometry used by the paired comfyui-krea2edit fork. Both are
valid trainers; they are not interchangeable.
If you want LoRAs that pair with the identity-edit inference stack, train with
`arch: "krea2_edit"` from this extension.

## Install

```bash
cd ai-toolkit/extensions
git clone https://github.com/chinoll/krea2edit-trainer krea2_edit
```

That's it — ai-toolkit discovers extensions in that folder. Set `arch: "krea2_edit"`
in your config. Python **3.10+** is required (the source uses PEP 604 `X | None`
annotations at runtime).

**This extension is CLI-only** (`python run.py <config>`): ai-toolkit's web UI job
builder has a hardcoded architecture list that does not include `krea2_edit`, so the
arch will not appear there. Write the YAML config and launch it from the command
line — and launch **from the ai-toolkit root**, because ai-toolkit resolves a config
path against its own `config/` folder and then the current directory:

```bash
cd ai-toolkit
python run.py extensions/krea2_edit/configs/krea2_edit_lora_512.yaml
```

## Model access

Krea 2 RAW weights are gated: accept the license at
[krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw), then point
`model.name_or_path` at the single-file `.safetensors` checkpoint (or a folder
containing one — the sharded diffusers layout is not supported). Setting it to the
repo id `krea/Krea-2-Raw` also works: the single-file checkpoint is downloaded for
you (set `HF_TOKEN` for the gated repo). Non-default filenames:
`model.model_kwargs.checkpoint_filename`.

## Dataset layout

Standard ai-toolkit paired-image datasets:

```
my_dataset/
  targets/          # edited result images + .txt captions (the edit instruction)
    0001.png
    0001.txt
  sources/          # reference images, stem-matched to targets
    0001.png
  sources_b/        # optional additional reference, stem-matched
    0001.png
```

- `folder_path` → `targets/`, `control_path` → `["sources/"]`. Add as many additional
  source folders as needed; every entry becomes a reference block in list order.
- **Every target needs a stem-matched source.** ai-toolkit pairs by filename stem
  only: for `targets/0001.png` it looks for `sources/0001.{jpg,jpeg,png,webp}`.
  Extensions may differ, stems may not. A target with no match keeps its caption,
  loses its reference, and **silently trains as plain T2I** — so an off-by-one naming
  slip degrades the run without any error. The trainer now audits the pairing at
  startup (filenames only) and refuses to start if any target is unpaired; it also
  warns loudly if a two-reference dataset has targets matched by only one of the two
  `control_path` folders (those items shift reference order — a target matched only
  by `sources_b/` gets it as reference #1).
- The caption is the *instruction* ("place her on a beach at sunset"), not a
  description of the target.
- Targets-only datasets (no `control_path`) train as plain T2I through the same arch —
  useful as regularization. Keep them as their own dataset entry; don't leave
  unpaired targets sitting in an edit dataset.
- **`cache_text_embeddings` must be `false`** to keep the per-step grounding jitter:
  text conditioning is image-grounded and re-jittered every step (see below), and a
  cached embedding freezes one grounding scale. The 24 GB tier deliberately trades
  that away — see the VRAM section.
- **Any number of references is supported.** Each `control_path` entry becomes a
  clean VAE token block and a Qwen3-VL vision block in the same order. The trainer
  always advertises multi-image support to ai-toolkit's cached-text path, so semantic
  grounding includes every configured reference without `model_kwargs.multi_ref`.
- **`flip_x` / `flip_y` must stay false** on edit datasets: ai-toolkit flips the
  target image but not the raw control (reference) images, so every flipped pair is
  silently desynced. The trainer raises on this when it can see the dataset config;
  pre-flip pairs offline (both images) if you want mirror augmentation.
- **`batch_size: 1`** — raw control images of mixed sizes cannot be collated (see
  "Not supported" below). Use `gradient_accumulation` for a larger effective batch.

## The recipe, in short

- **Grounded text encoding** — each reference image is fed to the Qwen3-VL text
  encoder alongside the instruction. Its image-patch hidden states are removed before
  the resulting context reaches the DiT, so the DiT receives only language-token states
  that have already been grounded by the VLM. Grounding resolution is jittered per step
  for scale robustness. **The released LoRAs trained at max 768 / jitter min 384**, which
  is also what this trainer now uses by default (and matches the inference node's
  `grounding_px` default of 768) — no env vars needed (run from the ai-toolkit root):
  ```bash
  python run.py extensions/krea2_edit/configs/krea2_edit_lora_512.yaml
  ```
  The env vars remain as *optional overrides*, e.g. a higher cap (which is **not**
  what the released weights used, and costs VRAM and step time):
  ```bash
  GROUNDING_MAX_PX=1024 GROUNDING_JITTER_MIN=384 python run.py extensions/krea2_edit/configs/krea2_edit_lora_512.yaml
  ```
  `GROUNDING_MAX_PX=0` disables the cap entirely (native-resolution grounding) — off
  the released recipe; the trainer warns when you do it.
  **Stale-cache caveat:** ai-toolkit's text-embedding cache key does not include the
  grounding settings. If you are training with `cache_text_embeddings: true` and you
  change `GROUNDING_MAX_PX` / `GROUNDING_JITTER_MIN`, delete the `_t_e_cache` folders
  in your dataset directories first — otherwise the run silently reuses embeddings
  built at the old grounding resolution.
- **Independent native reference grids** — each source keeps its own native aspect
  ratio and gets RoPE coordinates `(frame=i+1, h=0..Hr-1, w=0..Wr-1)`. It is never
  cropped, target-fitted, or center-offset. The only spatial adjustment is
  bottom/right replicate padding to a 16-pixel lattice for VAE/DiT patch alignment.
  This requires a newly trained LoRA; it is not compatible with released fit/crop
  weights.
- **Arbitrary target grids** — target H×W remains independent of every reference and
  is represented directly by Krea2's RoPE grid; set ai-toolkit's normal target
  resolution/bucket policy for training. At inference the paired
  `Krea2EditEmptyLatent` / `Krea2EditVAEDecode` nodes preserve any requested pixel
  output size, using and then removing bottom/right alignment padding only.
- **Separate reference time** — source latent tokens are clean and receive the DiT's
  `t=0` AdaLN modulation; noisy target tokens (and text) receive the sampled flow time.
  The implementation keeps the two modulation vectors as a small batch-concatenated
  pair, while attention still runs across the complete reference/target sequence.
- **Weighted flow-matching** — `timestep_type: "weighted"`: uniform timestep sampling
  with a per-timestep loss weight table.
- **Two-stage training works well**: a bulk skill/identity stage at 512, then a short
  finishing pass at 1024 warm-started from the best 512 checkpoint
  (`network.pretrained_lora_path`). Merging nearby checkpoints from the finishing
  stage often beats any single one.
- Rank: useful capacity saturates well below what you might expect — r64–128 is a
  good default; go higher only if you measure a reason to.
- **`model_kwargs: {checkpoint_every_n: N}`** — selective gradient checkpointing:
  with `train.gradient_checkpointing: true`, only every *N*th transformer block is
  recomputed instead of all of them, buying step time back for VRAM. `1` (the
  default) checkpoints every block — lowest VRAM, slowest. `2`–`4` is the useful
  range if you have headroom on the VRAM table below; the trainer prints how many
  blocks it ended up checkpointing at load.
- **`train.unload_text_encoder` must stay false.** The text encoder *is* the grounding
  path here; with it unloaded and nothing cached, ai-toolkit substitutes a blank
  embedding on every step and the run trains on no conditioning at all. The trainer
  raises at startup rather than let that happen.

### Not supported

These are refused with a clear error rather than silently training a LoRA the
inference nodes cannot reproduce:

- **In-training sample previews** for the edit arch. ai-toolkit's sampling path
  renders the inherited plain-T2I pipeline (no reference tokens, no image-grounded
  text encode), so previews would not show what the LoRA actually does. Use
  `train: { disable_sampling: true }` and evaluate checkpoints in ComfyUI.
- **`batch_size` > 1 with native reference grids.** ai-toolkit collates raw control images
  with `torch.cat`, which requires every source in a batch to have identical pixel
  dimensions; mixed-size datasets crash mid-run. Use `batch_size: 1` and raise
  `gradient_accumulation` instead.
- **Reference count is bounded by context length and VRAM.** Every reference adds a
  full VAE token grid and Qwen3-VL vision block. Start with a small number and avoid
  high-resolution references unless you have measured the resulting memory use.
- **Flip augmentation with native reference grids** (`flip_x` / `flip_y` on a dataset).
  ai-toolkit flips the target image but *not* the raw control images, silently
  desyncing every flipped pair. Keep both false, or pre-flip pairs offline (flipping
  target *and* source together, as a separate dataset folder).
- **Unpaired targets in an edit dataset.** A target with no stem-matched source trains
  as plain T2I with no warning; the startup audit refuses to start and names the
  offending files.
- **`train.unload_text_encoder: true`** — the grounding encode needs the text encoder
  resident; without it ai-toolkit trains on blank embeddings.

All of these are checked in the model constructor, i.e. before the base weights and
caches load, so a bad config fails in seconds rather than after a long warm-up.

## VRAM: measured requirements — read before filing an issue

These are **measured peak CUDA allocations** (fp8-quantized base + fp8 TE, batch 1,
gradient checkpointing, latent caching on, text embeddings NOT cached — i.e. the full
recipe with per-step grounding jitter, everything resident). Your card needs the peak
plus ~1–2 GB driver/display overhead.

| Config | Peak alloc | Fits on |
|---|---:|---|
| r64 @512 | **28.1 GB** | 32 GB cards, comfortably |
| r128 @512 | **32.0 GB** | does NOT fit 32 GB as-is — use 8-bit AdamW (~29 GB) or cached embeddings |
| r64 @768 | **30.9 GB** | 32 GB cards, tight (leave headroom: headless card recommended) |
| r16 @512, full bf16 (no quant) | 42.1 GB | reference worst case |

Field cross-check: independent beta testing on a 32 GB card (r128 @512, uncached)
read 31.3 GB and fell into PCIe offload — matching the table.

**What this means per card:**

- **32 GB**: the sweet spot. r64 @512 uncached runs the full recipe. r128 wants 8-bit
  AdamW or cached embeddings.
- **24 GB**: only with **cached text embeddings** (`cache_text_embeddings: true`),
  which evicts the 4 GB TE and lands ~24 GB — borderline, and caching freezes ONE
  grounding scale, trading away the scale-robustness the jitter provides. A
  multi-scale grounding cache that removes this tradeoff is planned for a follow-up
  release. The cache key ignores the grounding env vars — delete the dataset
  `_t_e_cache` folders after changing `GROUNDING_MAX_PX` / `GROUNDING_JITTER_MIN`.
- **16 GB**: **not supported today.** Please don't file issues asking why 1024
  training OOMs on 16 GB — the base model alone is 13 GB quantized. The planned
  grounding-cache mode plus aggressive settings may eventually enable 512px here.
- **1024px training**: not a consumer-card recipe. Expect it to exceed 32 GB
  even quantized. The shipped models did the bulk of their training at 512 and used
  1024 only for a short finishing pass — do that pass on rented hardware (an A6000/
  L40S-class 48 GB card or better).

**Troubleshooting install/env issues (found the hard way):**

- Install ai-toolkit's own `requirements.txt` exactly. A stale `diffusers` breaks
  aitk's *built-in* extensions with an opaque `Error running on_error` crash before
  this extension even loads.
- `adamw8bit` requires a bitsandbytes build matching your CUDA; if bnb prints a
  library-load error at startup it is harmless *unless* you selected an 8-bit
  optimizer — then switch to `adamw`.

Example configs are in `configs/`. Values in them (steps, repeats, learning rate)
are **generic starting points** — tune for your dataset.

## License / credits

Apache-2.0. `krea2_edit/src/` contains code vendored/adapted from the reference
implementation in [krea-ai/krea-2](https://github.com/krea-ai/krea-2) (Apache-2.0) —
see NOTICE. Krea 2 RAW **weights** are separately licensed under the Krea 2 Community
License (gated); this repo ships no weights.
