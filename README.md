# krea2edit trainer — ai-toolkit extension

*Community project — not affiliated with or endorsed by Krea.ai. "Krea" is used
descriptively to identify the base model this trainer targets.*

This is the actual training code behind the released
[krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit) LoRAs —
not a reimplementation. Its reference geometry is byte-matched to the
[comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) inference nodes
(v1.2.4+), so what you train is what the nodes run.

It adds two model architectures to [ai-toolkit](https://github.com/ostris/ai-toolkit):

- `krea2` — plain text-to-image LoRA training on Krea 2 RAW
- `krea2_edit` — instruction-based, reference-conditioned editing (the identity-edit
  recipe): in-context reference tokens + image-grounded text conditioning

## Install

```bash
cd ai-toolkit/extensions
git clone https://github.com/lbouaraba/krea2edit-trainer krea2_edit
```

That's it — ai-toolkit discovers extensions in that folder. Set `arch: "krea2_edit"`
(or `"krea2"`) in your config.

## Model access

Krea 2 RAW weights are gated: accept the license at
[krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw), then point
`model.name_or_path` at the checkpoint (single-file or diffusers layout).

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

## VRAM guidance (estimates — reports welcome)

| Tier | Setup | Notes |
|---|---|---|
| 24 GB | fp8 base (`quantize: true`), 512–768px, r64–128, 8-bit AdamW | closest to the shipped recipe |
| 32 GB+ | 1024px, higher rank affordable | full recipe |
| 16 GB | *planned* — needs a cached-variant grounding mode (tracked in issues) | not yet supported |

Example configs are in `configs/`. Values in them (steps, repeats, learning rate)
are **generic starting points** — tune for your dataset.

## License / credits

Apache-2.0. `krea2_edit/src/` contains code vendored/adapted from the reference
implementation in [krea-ai/krea-2](https://github.com/krea-ai/krea-2) (Apache-2.0) —
see NOTICE. Krea 2 RAW **weights** are separately licensed under the Krea 2 Community
License (gated); this repo ships no weights.
