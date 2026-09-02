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


def resize_to_size(image: torch.Tensor, height: int, width: int):
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


def threshold_alpha(image: Image.Image, alpha_transparency_threshold: int):
    if "A" not in image.getbands():
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A").point(
        lambda value: 0 if value <= alpha_transparency_threshold else value
    )
    rgba.putalpha(alpha)
    return rgba


def content_box(image: Image.Image):
    if "A" not in image.getbands():
        return 0, 0, image.width, image.height
    return image.getchannel("A").getbbox()


def flatten_alpha(image: Image.Image):
    if "A" not in image.getbands():
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def crop_and_flatten_alpha(
    image: Image.Image,
    min_pixels: int,
    alpha_transparency_threshold: int,
):
    image = threshold_alpha(image, alpha_transparency_threshold)
    box = content_box(image)
    if box is not None:
        image = image.crop(_expanded_crop_box(box, image.size, min_pixels))
    return flatten_alpha(image)


def image_to_tensor(image: Image.Image):
    return pil_to_tensor(flatten_alpha(image)).float().div_(255.0)


def load_pil_image(path: Path):
    with Image.open(path) as source:
        source.load()
        return source.copy()


def _normalized_box(box: tuple[int, int, int, int], image: Image.Image):
    left, top, right, bottom = box
    return (
        left / image.width,
        top / image.height,
        right / image.width,
        bottom / image.height,
    )


def _crop_box_from_normalized(
    box: tuple[float, float, float, float], image: Image.Image
):
    left, top, right, bottom = box
    left = min(image.width - 1, max(0, math.floor(left * image.width)))
    top = min(image.height - 1, max(0, math.floor(top * image.height)))
    right = max(left + 1, min(image.width, math.ceil(right * image.width)))
    bottom = max(top + 1, min(image.height, math.ceil(bottom * image.height)))
    return left, top, right, bottom


def crop_aligned_alpha_images(images: list[Image.Image], min_pixels: int):
    """Use one normalized alpha-content crop for target + source images."""
    boxes = []
    for image in images:
        box = content_box(image)
        if box is not None:
            boxes.append((box, image))
    if not boxes:
        return images

    normalized_boxes = [_normalized_box(box, image) for box, image in boxes]
    union_box = (
        min(box[0] for box in normalized_boxes),
        min(box[1] for box in normalized_boxes),
        max(box[2] for box in normalized_boxes),
        max(box[3] for box in normalized_boxes),
    )
    target_box = _crop_box_from_normalized(union_box, images[0])
    target_crop = _expanded_crop_box(target_box, images[0].size, min_pixels)
    normalized_crop = _normalized_box(target_crop, images[0])
    return [
        image.crop(_crop_box_from_normalized(normalized_crop, image))
        for image in images
    ]


def prepare_aligned_images(
    images: list[Image.Image],
    min_pixels: int,
    max_pixels: int,
    alpha_transparency_threshold: int,
):
    """Jointly crop RGBA pairs, then apply one uniform scale without distortion."""
    images = [
        threshold_alpha(image, alpha_transparency_threshold) for image in images
    ]
    if any("A" in image.getbands() for image in images):
        images = crop_aligned_alpha_images(images, min_pixels)

    areas = [image.width * image.height for image in images]
    lower_scale = (
        math.sqrt(min_pixels / min(areas)) if min_pixels > 0 else 0.0
    )
    upper_scale = (
        math.sqrt(max_pixels / max(areas)) if max_pixels > 0 else float("inf")
    )
    if lower_scale <= upper_scale:
        scale = min(max(1.0, lower_scale), upper_scale)
    else:
        # A shared scale cannot satisfy both bounds. Keep the group under the
        # maximum pixel budget rather than expanding a large source further.
        scale = upper_scale

    tensors = []
    for image in images:
        height = max(16, int(image.height * scale) // 16 * 16)
        width = max(16, int(image.width * scale) // 16 * 16)
        tensors.append(resize_to_size(image_to_tensor(image), height, width))
    return tensors


def prepare_image(
    source: Image.Image,
    min_pixels: int,
    max_pixels: int,
    alpha_transparency_threshold: int,
):
    if "A" in source.getbands():
        image = crop_and_flatten_alpha(
            source, min_pixels, alpha_transparency_threshold
        )
    else:
        image = source.convert("RGB")
    return resize_to_training_grid(
        image_to_tensor(image), min_pixels, max_pixels
    )


def load_image(
    path: Path,
    min_pixels: int,
    max_pixels: int,
    alpha_transparency_threshold: int,
):
    return prepare_image(
        load_pil_image(path),
        min_pixels,
        max_pixels,
        alpha_transparency_threshold,
    )


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
                                    "role": reference.get(
                                        "role", "source" if index == 0 else "reference"
                                    ),
                                }
                                for index, reference in enumerate(references)
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
        source_indices = [
            reference_index
            for reference_index, reference in enumerate(sample["references"])
            if reference["role"] == "source"
        ]
        references = [None] * len(sample["references"])

        if source_indices:
            aligned_images = prepare_aligned_images(
                [load_pil_image(sample["target"])]
                + [
                    load_pil_image(sample["references"][reference_index]["image"])
                    for reference_index in source_indices
                ],
                self.min_image_pixels,
                self.max_image_pixels,
                self.alpha_transparency_threshold,
            )
            target = aligned_images[0]
            for reference_index, image in zip(source_indices, aligned_images[1:]):
                references[reference_index] = image
        else:
            target = load_image(
                sample["target"],
                self.min_image_pixels,
                self.max_image_pixels,
                self.alpha_transparency_threshold,
            )

        for reference_index, reference in enumerate(sample["references"]):
            if references[reference_index] is None:
                references[reference_index] = load_image(
                    reference["image"],
                    self.min_image_pixels,
                    self.max_image_pixels,
                    self.alpha_transparency_threshold,
                )

        return {
            "id": sample["id"],
            "prompt": sample["prompt"],
            "target": target,
            "references": references,
            "source_indices": source_indices,
            "primary_source_index": source_indices[0] if source_indices else None,
        }


def ragged_collate(samples):
    return samples
