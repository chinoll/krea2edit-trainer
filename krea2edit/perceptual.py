"""Frozen feature distances used by decoded flow-matching objectives."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


DEFAULT_MODEL_PATHS = {
    "tipsv2": "google/tipsv2-b14",
    "dinov2": "facebook/dinov2-base",
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
}
DEFAULT_INPUT_RANGES = {
    # Bounds reflect global-image sizes seen while training/adapting the
    # released weights. They intentionally exclude smaller local-crop views
    # and are narrower than the architectures' forward limits.
    "tipsv2": (224, 448),
    "dinov2": (224, 518),
    "dinov3": (256, 768),
    # LPIPS is calibrated on 64px BAPPS patches over a VGG backbone trained at
    # 224px, so the composite metric's covered interval spans both stages.
    "vgg": (64, 224),
}
BACKBONE_ALIASES = {
    "dino2": "dinov2",
    "dino3": "dinov3",
}
SUPPORTED_PERCEPTUAL_BACKBONES = {"tipsv2", "dinov2", "dinov3", "vgg"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _finite_nonnegative(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")
    return result


def _canonical_backbone_name(name: str) -> str:
    name = str(name).lower()
    return BACKBONE_ALIASES.get(name, name)


def _size_pair(value, label: str) -> tuple[int, int]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        height = width = value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        dimensions = tuple(value)
        if len(dimensions) != 2:
            raise ValueError(f"{label} must contain exactly [height, width]")
        height, width = dimensions
    else:
        raise TypeError(f"{label} must be an integer or [height, width]")
    if any(
        not isinstance(dimension, Integral) or isinstance(dimension, bool)
        for dimension in (height, width)
    ):
        raise TypeError(f"{label} dimensions must be integers")
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} dimensions must be > 0")
    return height, width


def _align_up(size: tuple[int, int], patch: tuple[int, int]):
    return tuple(
        math.ceil(dimension / patch_dimension) * patch_dimension
        for dimension, patch_dimension in zip(size, patch)
    )


def _validate_range_bounds(
    minimum: tuple[int, int],
    maximum: tuple[int, int],
    label: str,
):
    if any(lower > upper for lower, upper in zip(minimum, maximum)):
        raise ValueError(
            f"{label} minimum {minimum} must not exceed maximum {maximum}"
        )


def _validate_size_range(
    size: tuple[int, int],
    minimum: tuple[int, int],
    maximum: tuple[int, int],
    label: str,
):
    _validate_range_bounds(minimum, maximum, label)
    if any(
        dimension < lower or dimension > upper
        for dimension, lower, upper in zip(size, minimum, maximum)
    ):
        raise ValueError(
            f"{label} {size} must be within [{minimum}, {maximum}]"
        )


def _validate_patch_alignment(
    size: tuple[int, int],
    patch: tuple[int, int],
    label: str,
):
    if any(
        dimension % patch_dimension != 0
        for dimension, patch_dimension in zip(size, patch)
    ):
        raise ValueError(
            f"{label} {size} must be divisible by patch grid {patch}"
        )


def _patch_grid_range(
    minimum: tuple[int, int],
    maximum: tuple[int, int],
    patch: tuple[int, int],
    label: str,
):
    grid_minimum = _align_up(minimum, patch)
    grid_maximum = tuple(
        dimension // patch_dimension * patch_dimension
        for dimension, patch_dimension in zip(maximum, patch)
    )
    if any(lower > upper for lower, upper in zip(grid_minimum, grid_maximum)):
        raise ValueError(
            f"{label} [{minimum}, {maximum}] contains no size aligned to "
            f"patch grid {patch}"
        )
    return grid_minimum, grid_maximum


def _fit_isotropic_size(
    size: tuple[int, int],
    minimum: tuple[int, int],
    maximum: tuple[int, int],
    grid_minimum: tuple[int, int],
    grid_maximum: tuple[int, int],
    patch: tuple[int, int],
    label: str,
):
    height, width = size
    min_height, min_width = grid_minimum
    max_height, max_width = grid_maximum
    minimum_scale = max(min_height / height, min_width / width)
    maximum_scale = min(max_height / height, max_width / width)
    if minimum_scale > maximum_scale:
        raise ValueError(
            f"{label} {size} cannot be isotropically resized into "
            f"patch-aligned range [{grid_minimum}, {grid_maximum}] "
            f"inside configured range [{minimum}, {maximum}]"
        )

    if minimum_scale > 1.0:
        scale = minimum_scale
    elif maximum_scale < 1.0:
        scale = maximum_scale
    else:
        scale = 1.0

    scaled = (height * scale, width * scale)
    resolved = tuple(
        min(
            max(
                math.floor(dimension / patch_dimension + 0.5)
                * patch_dimension,
                grid_lower,
            ),
            grid_upper,
        )
        for dimension, patch_dimension, grid_lower, grid_upper in zip(
            scaled, patch, grid_minimum, grid_maximum
        )
    )
    _validate_size_range(resolved, minimum, maximum, label)
    return resolved


def _feature_tokens(features: torch.Tensor):
    if features.ndim == 4:
        return features.flatten(2).transpose(1, 2)
    return features


class PerceptualFeatureLoss(nn.Module):
    """One frozen perceptual feature extractor.

    DINOv2 follows the public PFM all-layer normalized feature distance. DINOv3
    and TIPSv2 use the same all-layer distance over spatial patch tokens.
    """

    def __init__(
        self,
        name: str,
        model_path: str | None = None,
        input_size: int | Sequence[int] | None = None,
        min_input_size: int | Sequence[int] | None = None,
        max_input_size: int | Sequence[int] | None = None,
    ):
        super().__init__()
        self.name = _canonical_backbone_name(name)
        self.num_register_tokens = 0
        self.drop_cls_token = self.name == "dinov3"

        if self.name not in SUPPORTED_PERCEPTUAL_BACKBONES:
            choices = ", ".join(sorted(SUPPORTED_PERCEPTUAL_BACKBONES))
            raise ValueError(
                f"unsupported perceptual backbone {self.name!r}; "
                f"choose one of: {choices}"
            )

        default_minimum, default_maximum = DEFAULT_INPUT_RANGES[self.name]
        requested_size = (
            None
            if input_size is None
            else _size_pair(input_size, f"{self.name} input_size")
        )
        self.min_input_size = _size_pair(
            default_minimum if min_input_size is None else min_input_size,
            f"{self.name} min_input_size",
        )
        self.max_input_size = _size_pair(
            default_maximum if max_input_size is None else max_input_size,
            f"{self.name} max_input_size",
        )
        _validate_range_bounds(
            self.min_input_size,
            self.max_input_size,
            f"{self.name} input range",
        )
        if requested_size is not None:
            _validate_size_range(
                requested_size,
                self.min_input_size,
                self.max_input_size,
                f"{self.name} requested input_size",
            )

        if self.name == "vgg":
            try:
                from lpips import LPIPS
            except ImportError as error:
                raise ImportError(
                    "the VGG perceptual backbone requires the 'lpips' package"
                ) from error
            lpips_options = {}
            if model_path is not None:
                lpips_options["model_path"] = model_path
            self.model = LPIPS(
                net="vgg", pretrained=True, verbose=False, **lpips_options
            )
            patch_size = 1
        elif self.name == "dinov2":
            from transformers import Dinov2Model

            self.model = Dinov2Model.from_pretrained(
                model_path or DEFAULT_MODEL_PATHS[self.name],
                token=os.getenv("HF_TOKEN"),
            )
            patch_size = self.model.config.patch_size
            self.num_register_tokens = int(
                getattr(self.model.config, "num_register_tokens", 0)
            )
        elif self.name == "dinov3":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                model_path or DEFAULT_MODEL_PATHS[self.name],
                token=os.getenv("HF_TOKEN"),
            )
            patch_size = getattr(self.model.config, "patch_size", 1)
            self.num_register_tokens = int(
                getattr(self.model.config, "num_register_tokens", 0)
            )
        elif self.name == "tipsv2":
            from transformers import AutoModel

            tips_model = AutoModel.from_pretrained(
                model_path or DEFAULT_MODEL_PATHS[self.name],
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True,
            )
            patch_size = tips_model.config.patch_size
            self.model = tips_model.vision_encoder
            self.num_register_tokens = int(
                getattr(tips_model.config, "num_register_tokens", 0)
            )
            # Keep only the vision tower; the text tower is irrelevant to this loss.
            del tips_model
        self.patch_size = _size_pair(
            patch_size, f"{self.name} model patch_size"
        )
        self.grid_min_input_size, self.grid_max_input_size = _patch_grid_range(
            self.min_input_size,
            self.max_input_size,
            self.patch_size,
            f"{self.name} input range",
        )
        self.input_size = None
        if requested_size is not None:
            _validate_patch_alignment(
                requested_size,
                self.patch_size,
                f"{self.name} fixed input_size",
            )
            self.input_size = requested_size
        self.model.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        # The extractor is a fixed metric even while the DiT is training.
        super().train(False)
        self.model.eval()
        return self

    @staticmethod
    def _l2_normalize(features: torch.Tensor, eps: float = 1e-6):
        norm = torch.linalg.vector_norm(
            features, ord=2, dim=-1, keepdim=True
        ).clamp_min(eps)
        return features / norm

    def _resolve_input_size(self, image_size: tuple[int, int]):
        if self.input_size is not None:
            return self.input_size
        return _fit_isotropic_size(
            image_size,
            self.min_input_size,
            self.max_input_size,
            self.grid_min_input_size,
            self.grid_max_input_size,
            self.patch_size,
            f"{self.name} DiT output size",
        )

    @staticmethod
    def _resize(image: torch.Tensor, input_size: tuple[int, int]):
        return F.interpolate(
            image,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _prepare_imagenet(image: torch.Tensor):
        image = image.add(1.0).div(2.0)
        mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (image - mean) / std

    def _strip_special_tokens(self, features: torch.Tensor):
        if features.ndim != 3:
            return features
        if self.drop_cls_token:
            return features[:, 1 + self.num_register_tokens :]
        if self.num_register_tokens <= 0:
            return features
        return torch.cat(
            (
                features[:, :1],
                features[:, 1 + self.num_register_tokens :],
            ),
            dim=1,
        )

    def _transformer_states(self, image: torch.Tensor):
        outputs = self.model(
            pixel_values=self._prepare_imagenet(image),
            output_hidden_states=True,
        )
        states = outputs.hidden_states
        # DINOv2 returns embeddings followed by block outputs, while some
        # DINOv3 Transformers versions return block outputs only. Strip the
        # embedding state only when the model actually supplied one.
        num_hidden_layers = int(
            getattr(self.model.config, "num_hidden_layers", len(states))
        )
        if len(states) == num_hidden_layers + 1:
            states = states[1:]
        if self.name == "dinov3" and states:
            # Transformers releases disagree on whether DINOv3's final
            # hidden-state entry is before or after the final LayerNorm.
            # last_hidden_state is the stable, normalized public output.
            states = (*states[:-1], outputs.last_hidden_state)
        return tuple(
            self._strip_special_tokens(_feature_tokens(features))
            for features in states
        )

    def _tipsv2_states(self, image: torch.Tensor):
        # The public encode_image convenience method is no-grad. Calling the
        # frozen vision tower directly keeps gradients with respect to image.
        # Its intermediate-layer API normalizes every block output and removes
        # CLS/register tokens, leaving only spatial patch tokens.
        return tuple(
            self.model.get_intermediate_layers(
                image.add(1.0).div(2.0),
                n=self.model.n_blocks,
                reshape=False,
                return_class_token=False,
                norm=True,
            )
        )

    def _feature_states(self, image: torch.Tensor):
        if self.name == "tipsv2":
            return self._tipsv2_states(image)
        return self._transformer_states(image)

    def _resized_batch_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ):
        model_dtype = next(self.model.parameters()).dtype
        prediction = prediction.to(model_dtype)
        target = target.to(model_dtype)
        if self.name == "vgg":
            values = self.model(prediction, target)
            return values.reshape(values.shape[0], -1).mean(dim=1)

        prediction_states = self._feature_states(prediction)
        with torch.no_grad():
            target_states = self._feature_states(target)

        layer_losses = []
        for prediction_features, target_features in zip(
            prediction_states, target_states
        ):
            prediction_features = self._l2_normalize(prediction_features)
            target_features = self._l2_normalize(target_features)
            layer_losses.append(
                (prediction_features - target_features)
                .square()
                .mean(dim=tuple(range(1, prediction_features.ndim)))
            )
        return torch.stack(layer_losses, dim=0).mean(dim=0)

    def forward(
        self,
        predictions: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
    ):
        groups: dict[
            tuple[int, int],
            list[tuple[torch.Tensor, torch.Tensor]],
        ] = {}
        for prediction, target in zip(predictions, targets):
            input_size = self._resolve_input_size(
                tuple(prediction.shape[-2:])
            )
            groups.setdefault(input_size, []).append((prediction, target))

        sample_losses = []
        for input_size, pairs in groups.items():
            prediction_batch = torch.cat(
                [
                    self._resize(prediction.unsqueeze(0), input_size)
                    for prediction, _ in pairs
                ],
                dim=0,
            )
            target_batch = torch.cat(
                [
                    self._resize(target.unsqueeze(0), input_size)
                    for _, target in pairs
                ],
                dim=0,
            )
            sample_losses.append(
                self._resized_batch_loss(prediction_batch, target_batch)
            )
        return torch.cat(sample_losses, dim=0).mean()


class PerceptualLossEnsemble(nn.Module):
    """Absolute weighted sum of one or more frozen feature distances."""

    def __init__(
        self,
        backbones: list[dict],
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if not isinstance(backbones, list) or not backbones:
            raise ValueError(
                "train.loss.perceptual_backbones must be a non-empty list"
            )
        if not isinstance(gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean")

        modules = []
        weights = []
        for index, spec in enumerate(backbones):
            if not isinstance(spec, dict):
                raise TypeError(
                    f"train.loss.perceptual_backbones[{index}] must be a mapping"
                )
            unknown = set(spec) - {
                "name",
                "weight",
                "model_path",
                "input_size",
                "min_input_size",
                "max_input_size",
            }
            if unknown:
                raise ValueError(
                    f"unknown train.loss.perceptual_backbones[{index}] entries: "
                    + ", ".join(sorted(unknown))
                )
            if "name" not in spec:
                raise ValueError(
                    f"train.loss.perceptual_backbones[{index}].name is required"
                )
            weight = _finite_nonnegative(
                spec.get("weight", 1.0),
                f"train.loss.perceptual_backbones[{index}].weight",
            )
            if weight == 0.0:
                continue
            modules.append(
                PerceptualFeatureLoss(
                    name=spec["name"],
                    model_path=spec.get("model_path"),
                    input_size=spec.get("input_size"),
                    min_input_size=spec.get("min_input_size"),
                    max_input_size=spec.get("max_input_size"),
                )
            )
            weights.append(weight)

        if not modules:
            raise ValueError(
                "at least one perceptual backbone weight must be > 0"
            )
        self.backbones = nn.ModuleList(modules)
        self.weights = tuple(weights)
        self.gradient_checkpointing = gradient_checkpointing
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(False)
        for backbone in self.backbones:
            backbone.eval()
        return self

    def forward(
        self,
        predictions: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
    ):
        if len(predictions) != len(targets):
            raise ValueError(
                "perceptual prediction and target batches must contain the "
                f"same number of images, got {len(predictions)} and "
                f"{len(targets)}"
            )
        if len(predictions) == 0:
            raise ValueError("perceptual image batch must not be empty")
        for index, (prediction, target) in enumerate(
            zip(predictions, targets)
        ):
            if prediction.shape != target.shape:
                raise ValueError(
                    "perceptual x0 prediction and target must have identical "
                    f"shapes before resize at batch index {index}, got "
                    f"{tuple(prediction.shape)} and {tuple(target.shape)}"
                )

        total = predictions[0].new_zeros((), dtype=torch.float32)
        for weight, backbone in zip(self.weights, self.backbones):
            if self.gradient_checkpointing:
                image_count = len(predictions)

                def checkpointed_backbone(*images, module=backbone):
                    return module(
                        images[:image_count],
                        images[image_count:],
                    )

                backbone_loss = activation_checkpoint(
                    checkpointed_backbone,
                    *predictions,
                    *targets,
                    use_reentrant=False,
                )
            else:
                backbone_loss = backbone(predictions, targets)
            total = total + weight * backbone_loss.float()
        return total
