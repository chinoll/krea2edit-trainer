import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from krea2edit.mmdit import SingleMMDiTConfig, SingleStreamDiT


SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PROMPT_TEMPLATE_ENCODE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT_TEMPLATE_ENCODE_START_IDX = 34


KREA2_MMDIT_CONFIG = {
    "features": 6144,
    "tdim": 256,
    "txtdim": 2560,
    "heads": 48,
    "kvheads": 12,
    "multiplier": 4,
    "layers": 28,
    "patch": 2,
    "channels": 16,
    "txtheads": 20,
    "txtkvheads": 20,
    "txtlayers": 12,
}


def torch_dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def checkpoint_path(name_or_path: str, filename: str | None):
    path = Path(name_or_path)
    if path.is_file():
        return path
    if path.is_dir():
        return path / filename if filename else next(path.glob("*.safetensors"))
    filename = filename or f"{name_or_path.split('/')[-1].split('-')[-1].lower()}.safetensors"
    return Path(hf_hub_download(name_or_path, filename=filename, token=os.getenv("HF_TOKEN")))


def load_dit(config: dict, dtype: torch.dtype):
    architecture = KREA2_MMDIT_CONFIG | config.get("architecture", {})
    with torch.device("meta"):
        model = SingleStreamDiT(SingleMMDiTConfig(**architecture))
    state = load_file(str(checkpoint_path(config["name_or_path"], config.get("filename"))))
    state = {key: value.to(dtype) if value.is_floating_point() else value for key, value in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    model.name_or_path = config["name_or_path"]
    return model


def quantize_component(model: nn.Module, qtype_name: str):
    if qtype_name == "none":
        model.requires_grad_(False)
        return model
    from optimum.quanto import freeze, qfloat8, qint4, qint8, quantize

    qtype = {"int4": qint4, "int8": qint8, "float8": qfloat8}[qtype_name]
    quantize(model, weights=qtype)
    freeze(model)
    return model


def add_lora(model: SingleStreamDiT, config: dict):
    target_modules = config.get("target_modules", "all-linear")
    if target_modules == "all-linear":
        target_modules = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    lora = LoraConfig(
        r=int(config["rank"]),
        lora_alpha=int(config["alpha"]),
        lora_dropout=float(config.get("dropout", 0.0)),
        target_modules=target_modules,
        bias="none",
        init_lora_weights="gaussian",
    )
    return get_peft_model(model, lora)


class ConditioningModels:
    def __init__(self, config: dict, dtype: torch.dtype, device: torch.device):
        from diffusers import AutoencoderKLQwenImage
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        token = os.getenv("HF_TOKEN")
        self.processor = AutoProcessor.from_pretrained(config["text_encoder_path"], token=token)
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            config["text_encoder_path"], torch_dtype=dtype, token=token
        )
        quantize_component(self.text_encoder, config["quantization"]["text_encoder"])
        self.text_encoder.eval().to(device)

        self.vae = AutoencoderKLQwenImage.from_pretrained(
            config["vae_path"], subfolder="vae", torch_dtype=dtype, token=token
        )
        self.vae.requires_grad_(False).eval().to(device)
        self.dtype = dtype
        self.device = device
        self.grounding_max_px = int(config.get("grounding_max_px", 768))
        self.grounding_jitter_min = int(config.get("grounding_jitter_min", 384))

    def _grounding_pil(self, image: torch.Tensor):
        from torchvision.transforms.functional import to_pil_image

        cap = self.grounding_max_px
        if 0 < self.grounding_jitter_min < cap:
            cap = random.randint(self.grounding_jitter_min, cap)
        height, width = image.shape[-2:]
        if cap > 0 and max(height, width) > cap:
            scale = cap / max(height, width)
            image = F.interpolate(
                image.unsqueeze(0),
                size=(round(height * scale), round(width * scale)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return to_pil_image(image)

    @torch.no_grad()
    def encode_prompt(self, prompt: str, references: list[torch.Tensor]):
        images = [self._grounding_pil(image) for image in references]
        vision = "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
        text = PROMPT_TEMPLATE_ENCODE_PREFIX + vision + prompt + PROMPT_TEMPLATE_ENCODE_SUFFIX
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        output = self.text_encoder.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = torch.stack([output.hidden_states[index] for index in SELECT_LAYERS], dim=2)[0]
        input_ids = inputs["input_ids"][0]
        image_token_id = self.text_encoder.config.image_token_id
        hidden = hidden[PROMPT_TEMPLATE_ENCODE_START_IDX:]
        input_ids = input_ids[PROMPT_TEMPLATE_ENCODE_START_IDX:]
        return hidden[input_ids != image_token_id].to(self.dtype)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        pixels = image.to(self.device, self.dtype, non_blocking=True).mul(2).sub(1).unsqueeze(0).unsqueeze(2)
        latent = self.vae.encode(pixels).latent_dist.sample()
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.dtype)
        std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.dtype)
        latent = (latent - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)
        return latent[0, :, 0]


def latent_tokens(latent: torch.Tensor, patch: int, frame: int):
    _, height, width = latent.shape
    grid_h, grid_w = height // patch, width // patch
    tokens = rearrange(latent, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=patch, pw=patch)
    y, x = torch.meshgrid(
        torch.arange(grid_h, device=latent.device, dtype=torch.float32),
        torch.arange(grid_w, device=latent.device, dtype=torch.float32),
        indexing="ij",
    )
    positions = torch.stack(
        (torch.full_like(y, frame), y, x), dim=-1
    ).reshape(grid_h * grid_w, 3)
    return tokens, positions


class RaggedEditModel(nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit
        self.patch = dit.base_model.model.config.patch

    def forward(
        self,
        noisy_targets: list[torch.Tensor],
        references: list[list[torch.Tensor]],
        contexts: list[torch.Tensor],
        frames: list[list[int]],
        timesteps: torch.Tensor,
    ):
        patch = self.patch
        image_tokens, positions, reference_lengths = [], [], []
        for target, sample_refs, _, sample_frames in zip(
            noisy_targets, references, contexts, frames
        ):
            ref_tokens, ref_positions = [], []
            for latent, frame in zip(sample_refs, sample_frames):
                tokens, pos = latent_tokens(latent, patch, frame)
                ref_tokens.append(tokens)
                ref_positions.append(pos)
            target_tokens, target_positions = latent_tokens(target, patch, 0)
            image_tokens.append(torch.cat(ref_tokens + [target_tokens]))
            positions.append(torch.cat(ref_positions + [target_positions]))
            reference_lengths.append(sum(tokens.shape[0] for tokens in ref_tokens))

        if len(image_tokens) == 1:
            text_length = contexts[0].shape[0]
            text_positions = torch.zeros(
                (text_length, 3), dtype=positions[0].dtype, device=positions[0].device
            )
            mask = torch.ones(
                (1, text_length + image_tokens[0].shape[0]),
                dtype=torch.bool,
                device=image_tokens[0].device,
            )
            prediction = self.dit(
                img=image_tokens[0].unsqueeze(0),
                context=contexts[0].unsqueeze(0),
                t=timesteps,
                pos=torch.cat((text_positions, positions[0])).unsqueeze(0),
                mask=mask,
                ref_token_count=reference_lengths[0],
            )[0]
            return [prediction[reference_lengths[0]:]]

        return self.dit.forward_varlen(
            image_tokens=image_tokens,
            context=contexts,
            t=timesteps,
            image_positions=positions,
            ref_token_counts=reference_lengths,
        )


def target_velocity_tokens(clean: torch.Tensor, noise: torch.Tensor, patch: int):
    return latent_tokens(noise - clean, patch, 0)[0]
