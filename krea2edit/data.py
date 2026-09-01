import math
from pathlib import Path

import orjson
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm.auto import tqdm


def training_grid_size(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    alignment: int = 16,
):
    if min_pixels > 0 and height * width < min_pixels:
        scale = math.sqrt(min_pixels / float(height * width))
        height = max(1, math.ceil(height * scale))
        width = max(1, math.ceil(width * scale))
    if max_pixels > 0 and height * width > max_pixels:
        scale = math.sqrt(max_pixels / float(height * width))
        height = max(1, round(height * scale))
        width = max(1, round(width * scale))

    height = max(alignment, round(height / alignment) * alignment)
    width = max(alignment, round(width / alignment) * alignment)
    if min_pixels > 0 and height * width < min_pixels:
        scale = math.sqrt(min_pixels / float(height * width))
        height = max(alignment, math.ceil(height * scale / alignment) * alignment)
        width = max(alignment, math.ceil(width * scale / alignment) * alignment)
    if max_pixels > 0 and height * width > max_pixels:
        scale = math.sqrt(max_pixels / float(height * width))
        height = max(alignment, int(height * scale) // alignment * alignment)
        width = max(alignment, int(width * scale) // alignment * alignment)
    return height, width


def resize_to_training_grid(
    image: torch.Tensor,
    min_pixels: int,
    max_pixels: int,
    alignment: int = 16,
):
    """Aspect-preserving pixel range followed by a small resize to the model grid."""
    height, width = training_grid_size(
        *image.shape[-2:],
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        alignment=alignment,
    )

    if image.shape[-2:] != (height, width):
        image = F.interpolate(
            image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(0)
    return image


def _expanded_crop_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    min_pixels: int,
):
    left, top, right, bottom = box
    image_width, image_height = image_size
    content_width = right - left
    content_height = bottom - top
    target_area = min(
        max(min_pixels, content_width * content_height), image_width * image_height
    )
    if content_width * content_height >= target_area:
        return box

    aspect = content_width / content_height
    crop_width = min(
        image_width,
        max(content_width, math.ceil(math.sqrt(target_area * aspect))),
    )
    crop_height = min(
        image_height,
        max(content_height, math.ceil(target_area / crop_width)),
    )
    if crop_width * crop_height < target_area:
        crop_width = min(image_width, math.ceil(target_area / crop_height))

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    left = min(max(0, round(center_x - crop_width / 2.0)), image_width - crop_width)
    top = min(max(0, round(center_y - crop_height / 2.0)), image_height - crop_height)
    return left, top, left + crop_width, top + crop_height


def crop_and_flatten_alpha(
    image: Image.Image,
    min_pixels: int,
    alpha_transparency_threshold: int,
):
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A").point(
        lambda value: 0 if value <= alpha_transparency_threshold else value
    )
    rgba.putalpha(alpha)
    content_box = alpha.getbbox()
    if content_box is not None:
        rgba = rgba.crop(_expanded_crop_box(content_box, rgba.size, min_pixels))

    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def load_image(
    path: Path,
    min_pixels: int,
    max_pixels: int,
    alpha_transparency_threshold: int,
):
    with Image.open(path) as source:
        if "A" in source.getbands():
            image = crop_and_flatten_alpha(
                source, min_pixels, alpha_transparency_threshold
            )
        else:
            image = source.convert("RGB")
    tensor = pil_to_tensor(image).float().div_(255.0)
    return resize_to_training_grid(tensor, min_pixels, max_pixels)


class EditManifestDataset(Dataset):
    """JSONL dataset with one arbitrary-size target and N arbitrary-size references."""

    def __init__(
        self,
        manifest: str,
        min_image_pixels: int,
        max_image_pixels: int,
        alpha_transparency_threshold: int,
        show_progress: bool = True,
    ):
        self.manifest = Path(manifest).resolve()
        self.min_image_pixels = min_image_pixels
        self.max_image_pixels = max_image_pixels
        self.alpha_transparency_threshold = alpha_transparency_threshold
        self.samples = []
        manifest_progress = tqdm(
            total=self.manifest.stat().st_size,
            desc="Loading manifest metadata",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        pending_progress_bytes = 0
        with self.manifest.open("rb") as manifest_file:
            for line in manifest_file:
                if line.strip():
                    row = orjson.loads(line)
                    target = row["target"]
                    references = row["references"]
                    self.samples.append(
                        {
                            "id": row["id"],
                            "prompt": target.get("caption", target.get("prompt")),
                            "target": self._resolve(target["image"]),
                            "references": [
                                {
                                    "id": reference["id"],
                                    "image": self._resolve(reference["image"]),
                                }
                                for reference in references
                            ],
                        }
                    )
                pending_progress_bytes += len(line)
                if pending_progress_bytes >= 1024 * 1024:
                    manifest_progress.update(pending_progress_bytes)
                    pending_progress_bytes = 0
        manifest_progress.update(pending_progress_bytes)
        manifest_progress.set_postfix(samples=len(self.samples), refresh=False)
        manifest_progress.close()

    def _resolve(self, value: str):
        path = Path(value)
        return path if path.is_absolute() else (self.manifest.parent / path).resolve()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "id": sample["id"],
            "prompt": sample["prompt"],
            "target": load_image(
                sample["target"],
                self.min_image_pixels,
                self.max_image_pixels,
                self.alpha_transparency_threshold,
            ),
            "references": [
                load_image(
                    reference["image"],
                    self.min_image_pixels,
                    self.max_image_pixels,
                    self.alpha_transparency_threshold,
                )
                for reference in sample["references"]
            ],
        }


def ragged_collate(samples):
    return samples
