import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def training_grid_size(height: int, width: int, max_pixels: int, alignment: int = 16):
    if max_pixels > 0 and height * width > max_pixels:
        scale = math.sqrt(max_pixels / float(height * width))
        height = max(1, round(height * scale))
        width = max(1, round(width * scale))

    height = max(alignment, round(height / alignment) * alignment)
    width = max(alignment, round(width / alignment) * alignment)
    if max_pixels > 0 and height * width > max_pixels:
        scale = math.sqrt(max_pixels / float(height * width))
        height = max(alignment, int(height * scale) // alignment * alignment)
        width = max(alignment, int(width * scale) // alignment * alignment)
    return height, width


def resize_to_training_grid(image: torch.Tensor, max_pixels: int, alignment: int = 16):
    """Aspect-preserving pixel cap followed by a small resize to the model grid."""
    height, width = training_grid_size(
        *image.shape[-2:], max_pixels=max_pixels, alignment=alignment
    )

    if image.shape[-2:] != (height, width):
        image = F.interpolate(
            image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(0)
    return image


def load_image(path: Path, max_pixels: int):
    image = Image.open(path).convert("RGB")
    tensor = pil_to_tensor(image).float().div_(255.0)
    return resize_to_training_grid(tensor, max_pixels)


class EditManifestDataset(Dataset):
    """JSONL dataset with one arbitrary-size target and N arbitrary-size references."""

    def __init__(self, manifest: str, max_image_pixels: int):
        self.manifest = Path(manifest).resolve()
        self.max_image_pixels = max_image_pixels
        self.samples = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
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
            "target": load_image(sample["target"], self.max_image_pixels),
            "references": [
                load_image(reference["image"], self.max_image_pixels)
                for reference in sample["references"]
            ],
        }


def ragged_collate(samples):
    return samples
