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

Every raw image entering the edit-conditioning branch is limited independently by
`model_kwargs.max_image_pixels`
(default: `1048576`, i.e. 1 MP). If `H × W` exceeds the limit, it is downscaled by
`sqrt(limit / (H × W))`, preserving aspect ratio, and then aligned to the 16px
VAE/DiT grid. The final alignment rounds down when needed, so the pixel limit remains
a real ceiling. Set `0` to disable the cap.

The limit is applied before both VAE encoding and Qwen3-VL grounding during training.
ComfyUI inference deliberately keeps its existing arbitrary-size, 16px-aligned input
behavior and does not expose this training-only cap.

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

## Dataset layout

ai-toolkit pairs target and reference files by stem:

```text
data/my_dataset/
├── targets/
│   ├── 0001.png
│   └── 0001.txt        # edit instruction, not target-image caption
├── sources/
│   └── 0001.jpg
└── sources_b/          # optional second reference family
    └── 0001.png
```

Every `control_path` becomes one ordered reference block. References may have
different dimensions and a sample may have a different number of matched references,
subject to available context length and VRAM.

## Minimal configuration

```yaml
datasets:
  - folder_path: data/my_dataset/targets
    control_path: [data/my_dataset/sources, data/my_dataset/sources_b]
    caption_ext: txt
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
within each batch; it does not resize conditioning images to targets. Configure the
dataset resolution/buckets separately to bound target-image pixels.

## Important constraints

- Target latents are still dense `(B, C, H, W)`, because ai-toolkit's scheduler and
  flow-matching loss are dense. Fully ragged target output sizes in one batch are not
  supported.
- Do not mix edit examples and targets-only/T2I examples in the same batch. Keep them
  in separate datasets or batches.
- Keep `flip_x` and `flip_y` disabled for edit datasets. ai-toolkit augments targets
  but does not apply the identical flip to raw references.
- Keep `train.unload_text_encoder: false`; Qwen3-VL is the grounding path.
- In-training previews are disabled. Use the paired ComfyUI nodes to evaluate saved
  LoRAs with the actual reference-conditioned architecture.

## Compatibility

This project is intentionally a separate architecture from the upstream trainer and
from ai-toolkit's built-in Krea 2 edit mode. Do not expect its LoRAs, position layout,
or reference geometry to interchange with this repository.
