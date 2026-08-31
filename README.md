# Krea2Edit Trainer — Ragged Multi-Reference Architecture

This is a **new training architecture**, not a small patch to the original
Krea2Edit trainer. It is designed for in-context image editing with independent,
multi-reference inputs and variable reference resolutions. Checkpoints trained with
this repository are **not compatible** with the released target-fitted/cropped
Krea2 identity-edit LoRAs; train a new LoRA for this architecture.

Upstream references:

- Original trainer: [lbouaraba/krea2edit-trainer](https://github.com/lbouaraba/krea2edit-trainer)
- Training framework: [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)

The upstream repositories do not provide this exact training contract. This fork
keeps ai-toolkit's target/noise/loss loop, but replaces the edit-conditioning path
with the architecture described below.

## Architecture

```text
reference images + instruction
        │
        ├─ Qwen3-VL ──► remove visual hidden states ──► grounded language tokens
        │
        └─ per-reference VAE encode ──► clean reference latent tokens

[grounded text | ref_1 | ... | ref_N | noisy target]
        │
        └─ packed DiT attention
             ├─ reference blocks: frame = 1..N, timestep = 0
             ├─ target block:     frame = 0,    timestep = sampled t
             └─ text block:       position = (0, 0, 0), timestep = sampled t
```

The VLM sees the source images, but its visual-token hidden states are removed before
the DiT. Therefore visual semantics reach the DiT through image-grounded language
states, while source appearance and layout reach it through the clean VAE latent
reference blocks.

Each reference starts a separate RoPE grid:

```text
ref_i: (frame=i, h=0..H_i-1, w=0..W_i-1)
target: (frame=0, h=0..H_t-1, w=0..W_t-1)
```

There is no crop, target fitting, or coordinate offset for references.

## Ragged reference batches

`batch_size > 1` is supported when reference images differ in resolution or number.
The extension replaces ai-toolkit's dense raw-control collation with a per-sample
reference list, VAE-encodes every reference independently, then packs each sample's
`[text | refs | target]` sequence using `cu_seqlens`.

The main DiT attention automatically selects an unpadded varlen backend:

| GPU | Backend | Install |
| --- | --- | --- |
| A100 / A800 / other Ampere or Ada | FlashAttention-2 | `pip install flash-attn` |
| H100 / H200 / Blackwell | FlashAttention-4 preferred | `pip install flash-attn-4` |

On Hopper and newer, FA2 remains a varlen fallback if FA4 is unavailable and the
installed FA2 build supports that GPU. There is intentionally no padding+SDPA fallback
for ragged B>1 batches.

`batch_size: 1` retains the regular PyTorch SDPA path and does not require either
FlashAttention package.

## Per-image pixel budget

Every training image is limited independently by `model_kwargs.max_image_pixels`
(default: `1048576`, i.e. 1 MP). If `H × W` exceeds the limit, it is downscaled by
`sqrt(limit / (H × W))`, preserving aspect ratio, and then aligned to the 16px
VAE/DiT grid. The final alignment rounds down when needed, so the pixel limit remains
a real ceiling. Set `0` to disable the cap.

For target images, the extension scales the raw target tensor (or a cached target
latent, using its equivalent pixel area) during collation, before noise and loss are
formed. Conditioning images use the same cap before both VAE encoding and Qwen3-VL
grounding. ComfyUI inference deliberately keeps its existing arbitrary-size,
16px-aligned input behavior and does not expose this training-only cap.

## Install

Install a current [ai-toolkit](https://github.com/ostris/ai-toolkit), then clone this
repository into its extension directory:

```bash
cd ai-toolkit/extensions
git clone https://github.com/chinoll/krea2edit-trainer krea2_edit
```

Install the FlashAttention package appropriate for the training GPU when using
`batch_size > 1` ragged references. Launch from the ai-toolkit root:

```bash
python run.py extensions/krea2_edit/configs/krea2_edit_lora_512.yaml
```

Python 3.10+ is required. Krea 2 RAW is gated; accept its license and provide a valid
Hugging Face token if needed.

## Manifest-first dataset format

The loader's source of truth is a JSONL manifest, not folder names or matching file
stems. Set `dataset_path` to this `.jsonl` file and do **not** set `control_path`.
The extension adapts it at runtime through ai-toolkit's public dataset objects; it
does not require editing ai-toolkit's source tree.

```jsonl
{"id":"edit-0001","target":{"image":"targets/0001.png","caption":"Place the subject on a beach at sunset."},"references":[{"id":"scene","frame":1,"image":"refs/0001_scene.jpg"},{"id":"subject","frame":2,"image":"refs/0001_subject.png"}]}
{"id":"edit-0002","target":{"image":"targets/0002.jpg","caption":"Make the jacket red."},"references":[{"id":"subject","frame":2,"image":"refs/0002_subject.jpg"}]}
```

All paths are relative to the manifest file (absolute paths also work). Each row has:

- `id`: stable sample ID for logs and auditing.
- `target.image`: target-image path.
- `target.caption` (or `target.prompt`): inline edit instruction.
- `references`: ordered by explicit positive integer `frame`; every reference has a
  stable semantic `id`, `frame`, and `image` path.

Reference frames do not get renumbered when an earlier role is absent: in the second
example, `subject` remains frame 2. The loader validates duplicate targets, reference
IDs, frames, paths, malformed JSON, and mixed edit/T2I rows before model weights load.
The extension writes a deterministic caption-map cache under `.krea2edit_cache/`;
ai-toolkit then uses its normal image, bucket, latent-cache, and text-cache mechanisms.

See [krea2_edit_manifest.example.jsonl](configs/krea2_edit_manifest.example.jsonl).

## Minimal configuration

```yaml
datasets:
  - dataset_path: data/my_dataset/manifest.jsonl
    resolution: [512]
    buckets: true
    cache_latents_to_disk: true
    cache_text_embeddings: false

train:
  batch_size: 2
  gradient_checkpointing: true
  unload_text_encoder: false
  disable_sampling: true

model:
  name_or_path: krea/Krea-2-Raw
  arch: krea2_edit
  model_kwargs:
    text_encoder_path: Qwen/Qwen3-VL-4B-Instruct
    max_image_pixels: 1048576
```

`buckets: true` is required for ragged B>1 edit batches. It aligns **target** H×W
within each batch; the same per-image cap is applied to every target before the batch
is stacked. It does not resize conditioning images to target geometry.

## In-training edit previews

Previews use the real edit path: reference images are encoded by both Qwen3-VL and
the VAE, reference DiT tokens remain clean at `t=0`, and only the target starts from
noise. AI-Toolkit still controls `sample_every`, seeds, saving, and logging.

The recommended setup reuses the training manifest. Add its path to `model_kwargs`,
then put manifest sample IDs (not captions) in `sample.samples[].prompt`:

```yaml
train:
  disable_sampling: false

model:
  model_kwargs:
    text_encoder_path: Qwen/Qwen3-VL-4B-Instruct
    max_image_pixels: 1048576
    preview_manifest: data/my_dataset/manifest.jsonl

sample:
  sample_every: 250
  sample_steps: 20
  guidance_scale: 4.5
  samples:
    - prompt: scene-subject-0001
      width: 512
      height: 512
      seed: 42
    - prompt: subject-only-0002
      width: 512
      height: 768
      seed: 43
```

This supports every reference count and preserves the manifest's explicit frame IDs.
For a quick preview without a manifest, `ctrl_img_1`, `ctrl_img_2`, and `ctrl_img_3`
also work, with contiguous frames `1..N`.

Each saved sample is a comparison sheet: the edit prompt is rendered above
`[input reference montage | output image]`. Keep seeds fixed to compare the same
examples across checkpoints.

## Important constraints

- Target latents are still dense `(B, C, H, W)`, because ai-toolkit's scheduler and
  flow-matching loss are dense. Fully ragged target output sizes in one batch are not
  supported.
- Do not mix edit examples and targets-only/T2I examples in the same batch. Keep them
  in separate datasets or batches.
- Keep `flip_x` and `flip_y` disabled for edit datasets. ai-toolkit augments targets
  but does not apply the identical flip to raw references.
- Keep `train.unload_text_encoder: false`; Qwen3-VL is the grounding path.
- Preview prompt embeddings are recomputed from the reference images at every sample
  event; cached text-only sample embeddings are intentionally not used.

## Compatibility

This project is intentionally a separate architecture from the upstream trainer and
from ai-toolkit's built-in Krea 2 edit mode. Do not expect its LoRAs, position layout,
or reference geometry to interchange with this repository.
