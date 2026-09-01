import math
import re
import textwrap
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms.functional import to_pil_image

from krea2edit.data import training_grid_size


class PreviewDatasetView:
    """ID-addressable view that only indexes the configured preview samples."""

    def __init__(self, dataset, sample_ids: list[str]):
        self.dataset = dataset
        self.max_image_pixels = dataset.max_image_pixels
        wanted = set(sample_ids)
        self.indices_by_id = {}
        if not wanted:
            return
        for index, sample in enumerate(dataset.samples):
            if sample["id"] in wanted:
                self.indices_by_id[sample["id"]] = index
                if len(self.indices_by_id) == len(wanted):
                    break

    def get_by_id(self, sample_id: str):
        return self.dataset[self.indices_by_id[sample_id]]


def flow_timesteps(
    token_count: int,
    steps: int,
    min_tokens: int = 256,
    max_tokens: int = 6400,
    min_shift: float = 0.5,
    max_shift: float = 1.15,
    mu: float | None = None,
):
    """Krea resolution-aware flow schedule, ordered from pure noise to clean."""
    if mu is None:
        slope = (max_shift - min_shift) / (max_tokens - min_tokens)
        mu = slope * token_count + (min_shift - slope * min_tokens)
    shift = math.exp(mu)
    timesteps = torch.linspace(1.0, 0.0, steps + 1)
    return (shift * timesteps / (1.0 + (shift - 1.0) * timesteps)).tolist()


def velocity_tokens_to_latent(
    tokens: torch.Tensor,
    latent_height: int,
    latent_width: int,
    patch: int,
):
    channels = tokens.shape[-1] // (patch * patch)
    return rearrange(
        tokens,
        "(h w) (c ph pw) -> c (h ph) (w pw)",
        h=latent_height // patch,
        w=latent_width // patch,
        c=channels,
        ph=patch,
        pw=patch,
    )


@torch.inference_mode()
def sample_edit(
    model,
    conditioning,
    prompt: str,
    references: list[torch.Tensor],
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    max_image_pixels: int,
    negative_prompt: str = "",
    schedule_mu: float | None = None,
):
    alignment = conditioning.vae_scale_factor * model.patch
    height, width = training_grid_size(
        height, width, max_pixels=max_image_pixels, alignment=alignment
    )
    latent_width = width // conditioning.vae_scale_factor
    latent_height = height // conditioning.vae_scale_factor

    conditional_context = conditioning.encode_prompt(
        prompt, references, grounding_jitter=False
    )
    unconditional_context = None
    if guidance_scale > 0:
        unconditional_context = conditioning.encode_prompt(
            negative_prompt, references, grounding_jitter=False
        )
    reference_latents = [
        conditioning.encode_image(image, sample_posterior=False)
        for image in references
    ]

    generator = torch.Generator(device=conditioning.device).manual_seed(seed)
    channels = model.dit.base_model.model.config.channels
    latent = torch.randn(
        (channels, latent_height, latent_width),
        generator=generator,
        device=conditioning.device,
        dtype=torch.float32,
    )
    token_count = (latent_height // model.patch) * (latent_width // model.patch)
    timesteps = flow_timesteps(token_count, steps, mu=schedule_mu)

    for current_t, previous_t in zip(timesteps[:-1], timesteps[1:]):
        timestep = torch.tensor(
            [current_t], device=conditioning.device, dtype=conditioning.dtype
        )
        conditional_tokens = model(
            [latent.to(conditioning.dtype)],
            [reference_latents],
            [conditional_context],
            timestep,
        )[0]
        velocity = velocity_tokens_to_latent(
            conditional_tokens, latent_height, latent_width, model.patch
        )
        if unconditional_context is not None:
            unconditional_tokens = model(
                [latent.to(conditioning.dtype)],
                [reference_latents],
                [unconditional_context],
                timestep,
            )[0]
            unconditional_velocity = velocity_tokens_to_latent(
                unconditional_tokens, latent_height, latent_width, model.patch
            )
            velocity = velocity + guidance_scale * (
                velocity - unconditional_velocity
            )
        latent = latent + (previous_t - current_t) * velocity.float()

    return to_pil_image(conditioning.decode_image(latent).cpu())


def render_preview_sheet(
    references: list[torch.Tensor],
    output: Image.Image,
    target: torch.Tensor,
    prompt: str,
):
    """Render the prompt above ``[reference montage | generated | ground truth]``."""
    output = output.convert("RGB")
    panel_width, panel_height = output.size
    input_panel = Image.new("RGB", output.size, (24, 24, 24))
    columns = max(1, math.ceil(math.sqrt(len(references))))
    rows = math.ceil(len(references) / columns)
    cell_width = panel_width // columns
    cell_height = panel_height // rows

    for index, reference in enumerate(references):
        image = to_pil_image(reference.detach().cpu().clamp(0, 1)).convert("RGB")
        image = ImageOps.contain(image, (cell_width, cell_height))
        x = (index % columns) * cell_width + (cell_width - image.width) // 2
        y = (index // columns) * cell_height + (cell_height - image.height) // 2
        input_panel.paste(image, (x, y))

    target_image = to_pil_image(target.detach().cpu().clamp(0, 1)).convert("RGB")
    target_image = ImageOps.contain(target_image, (panel_width, panel_height))
    target_panel = Image.new("RGB", output.size, (24, 24, 24))
    target_panel.paste(
        target_image,
        (
            (panel_width - target_image.width) // 2,
            (panel_height - target_image.height) // 2,
        ),
    )

    body = Image.new("RGB", (panel_width * 3, panel_height), "black")
    body.paste(input_panel, (0, 0))
    body.paste(output, (panel_width, 0))
    body.paste(target_panel, (panel_width * 2, 0))

    font_size = max(18, min(32, body.width // 48))
    font = ImageFont.load_default(size=font_size)
    lines = textwrap.wrap(
        str(prompt), width=max(24, int(body.width / (font_size * 0.6)))
    ) or [""]
    header_height = 20 + (font_size + 6) * len(lines)
    sheet = Image.new("RGB", (body.width, header_height + body.height), "white")
    ImageDraw.Draw(sheet).multiline_text(
        (12, 10), "\n".join(lines), fill="black", font=font, spacing=4
    )
    sheet.paste(body, (0, header_height))
    return sheet


def generate_previews(
    model,
    conditioning,
    dataset,
    config: dict,
    output_dir: Path,
    step: int,
):
    step_dir = output_dir / "samples" / f"step-{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    results = []
    was_training = model.training
    model.eval()
    try:
        for specification in config["samples"]:
            sample = dataset.get_by_id(specification["id"])
            height = int(specification.get("height", sample["target"].shape[-2]))
            width = int(specification.get("width", sample["target"].shape[-1]))
            seed = int(specification.get("seed", 42))
            output = sample_edit(
                model=model,
                conditioning=conditioning,
                prompt=sample["prompt"],
                references=sample["references"],
                width=width,
                height=height,
                steps=int(config["steps"]),
                guidance_scale=float(config["guidance_scale"]),
                seed=seed,
                max_image_pixels=dataset.max_image_pixels,
                negative_prompt=str(config.get("negative_prompt", "")),
                schedule_mu=config.get("schedule_mu"),
            )
            sheet = render_preview_sheet(
                sample["references"], output, sample["target"], sample["prompt"]
            )
            filename = re.sub(r"[^0-9A-Za-z._-]+", "_", sample["id"])
            path = step_dir / f"{filename}.webp"
            sheet.save(path, format="WEBP", quality=80)
            results.append((sample["id"], path))
    finally:
        model.train(was_training)
    return results
