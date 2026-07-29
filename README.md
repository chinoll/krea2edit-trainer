# krea2edit trainer — ai-toolkit extension

*Community project — not affiliated with or endorsed by Krea.ai. "Krea" is used
descriptively to identify the base model this trainer targets.*

This is the actual training code behind the released
[krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit) LoRAs —
not a reimplementation. Its reference geometry is byte-matched to the
[comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) inference nodes
(v1.2.4+), so what you train is what the nodes run.

It adds one model architecture to [ai-toolkit](https://github.com/ostris/ai-toolkit):

- `krea2_edit` — instruction-based, reference-conditioned editing (the identity-edit
  recipe): in-context reference tokens + image-grounded text conditioning

(Plain Krea 2 text-to-image training is already built into upstream ai-toolkit as
`arch: "krea2"` — this extension doesn't duplicate it.)

**Note on upstream's own edit mode:** ai-toolkit's built-in `krea2` arch also offers
an edit mode (`model_kwargs: {edit: true}`). It is a *different training contract* —
a "Picture N:"-labeled grounding template and an area-budget reference resize —
whereas this extension implements the exact grounding template and fit-to-target-grid
reference geometry that the comfyui-krea2edit nodes (and the released
krea2-identity-edit LoRAs) use. Both are valid trainers; they are not interchangeable.
If you want LoRAs that pair with the identity-edit inference stack, train with
`arch: "krea2_edit"` from this extension.

## Install

```bash
cd ai-toolkit/extensions
git clone https://github.com/lbouaraba/krea2edit-trainer krea2_edit
```

That's it — ai-toolkit discovers extensions in that folder. Set `arch: "krea2_edit"`
in your config.

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
  sources_b/        # optional second reference (multi-ref), stem-matched
    0001.png
```

- `folder_path` → `targets/`, `control_path` → `["sources/"]` (add `sources_b/` for
  two-reference training; references appear in the token sequence in list order).
- The caption is the *instruction* ("place her on a beach at sunset"), not a
  description of the target.
- Targets-only datasets (no `control_path`) train as plain T2I through the same arch —
  useful as regularization.
- **`cache_text_embeddings: false` is required** for edit datasets: text conditioning
  is image-grounded and jittered per step (see below), so it must not be cached.

## The recipe, in short

- **Grounded text encoding** — each reference image is fed to the Qwen3-VL text
  encoder alongside the instruction. Grounding resolution is jittered per step for
  scale robustness; control it with env vars at launch:
  ```bash
  GROUNDING_MAX_PX=1024 GROUNDING_JITTER_MIN=384 python run.py config/your_config.yaml
  ```
- **Fit reference protocol** (`fit` geometry) — references are AR-preserving fitted to
  the target grid with exact /16 alignment and fractionally-centered positions. This
  matches the v1.2.4+ node geometry exactly; older reimplementations that floor to
  /16 or use integer offsets produce seam artifacts.
- **Weighted flow-matching** — `timestep_type: "weighted"`: uniform timestep sampling
  with a per-timestep loss weight table.
- **Two-stage training works well**: a bulk skill/identity stage at 512, then a short
  finishing pass at 1024 warm-started from the best 512 checkpoint
  (`network.pretrained_lora_path`). Merging nearby checkpoints from the finishing
  stage often beats any single one.
- Rank: useful capacity saturates well below what you might expect — r64–128 is a
  good default; go higher only if you measure a reason to.

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
  release.
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
