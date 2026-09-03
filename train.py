#!/usr/bin/env python
import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torch.func import functional_call
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from tqdm.auto import tqdm

from krea2edit.data import EditManifestDataset, ragged_collate
from krea2edit.ema import TrainableParameterEMA
from krea2edit.modeling import (
    ConditioningModels,
    RaggedEditModel,
    add_lora,
    checkpoint_path,
    export_comfyui_lora,
    load_dit,
    quantize_component,
    target_velocity_tokens,
    torch_dtype,
)
from krea2edit.perceptual import PerceptualLossEnsemble
from krea2edit.svdquant import (
    convert_forward_svdquant_checkpoint,
    load_dual_svdquant_linears,
)
from krea2edit.sampling import (
    PreviewDatasetView,
    generate_previews,
    velocity_tokens_to_latent,
)
from krea2edit.timesteps import sample_timesteps


DEEPSPEED_MUON_NAMES = {"deepspeed_muon", "muon"}
TRAINING_BRANCHES = ("edit", "instruction_dropout", "noop")
VAE_LATENT_STRATEGIES = {"mode", "independent_sample"}
LOSS_NAMES = ("fm", "perceptual", "pixel", "self_flow")
PIXEL_LOSS_TYPES = {"mse", "l1", "huber", "charbonnier"}
DEFAULT_PERCEPTUAL_BACKBONES = [
    {
        "name": "dinov2",
        "model_path": "facebook/dinov2-base",
        "input_size": 518,
        "min_input_size": 224,
        "max_input_size": 518,
        "weight": 1.0,
    },
    {
        "name": "vgg",
        "input_size": 224,
        "min_input_size": 64,
        "max_input_size": 224,
        "weight": 1.0,
    },
]


def ema_recipe(config: dict):
    raw_ema = config.get("ema", {})
    if not isinstance(raw_ema, dict):
        raise TypeError("train.ema must be a mapping")
    allowed_entries = {
        "enabled",
        "decay",
        "warmup_steps",
        "update_every",
        "device",
    }
    unknown_entries = set(raw_ema) - allowed_entries
    if unknown_entries:
        raise ValueError(
            "unknown train.ema entries: " + ", ".join(sorted(unknown_entries))
        )

    enabled = raw_ema.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("train.ema.enabled must be a boolean")
    decay = float(raw_ema.get("decay", 0.9999))
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("train.ema.decay must be finite and in [0, 1)")
    warmup_steps = raw_ema.get("warmup_steps", 0)
    update_every = raw_ema.get("update_every", 1)
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
    ):
        raise ValueError("train.ema.warmup_steps must be an integer >= 0")
    if (
        isinstance(update_every, bool)
        or not isinstance(update_every, int)
        or update_every <= 0
    ):
        raise ValueError("train.ema.update_every must be an integer > 0")
    device = str(raw_ema.get("device", "accelerator")).lower()
    if device not in {"cpu", "accelerator"}:
        raise ValueError("train.ema.device must be cpu or accelerator")
    return {
        "enabled": enabled,
        "decay": decay,
        "warmup_steps": warmup_steps,
        "update_every": update_every,
        "device": device,
    }


def self_flow_recipe(config: dict):
    raw = config.get("self_flow", {})
    if not isinstance(raw, dict):
        raise TypeError("train.self_flow must be a mapping")
    allowed_entries = {
        "enabled",
        "mask_ratio",
        "student_layer_ratio",
        "teacher_layer_ratio",
        "student_layer",
        "teacher_layer",
        "projection",
        "projection_hidden_dim",
    }
    unknown_entries = set(raw) - allowed_entries
    if unknown_entries:
        raise ValueError(
            "unknown train.self_flow entries: "
            + ", ".join(sorted(unknown_entries))
        )

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("train.self_flow.enabled must be a boolean")
    mask_ratio = float(raw.get("mask_ratio", 0.25))
    if not math.isfinite(mask_ratio) or not 0.0 <= mask_ratio <= 0.5:
        raise ValueError("train.self_flow.mask_ratio must be finite and in [0, 0.5]")

    student_layer_ratio = float(raw.get("student_layer_ratio", 0.3))
    teacher_layer_ratio = float(raw.get("teacher_layer_ratio", 0.7))
    for name, value in (
        ("student_layer_ratio", student_layer_ratio),
        ("teacher_layer_ratio", teacher_layer_ratio),
    ):
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"train.self_flow.{name} must be finite and in (0, 1]")

    explicit_layers = {}
    for name in ("student_layer", "teacher_layer"):
        value = raw.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"train.self_flow.{name} must be an integer > 0")
        explicit_layers[name] = value

    projection = str(raw.get("projection", "mlp")).lower()
    if projection not in {"mlp", "identity"}:
        raise ValueError("train.self_flow.projection must be mlp or identity")
    projection_hidden_dim = raw.get("projection_hidden_dim", 1024)
    if (
        isinstance(projection_hidden_dim, bool)
        or not isinstance(projection_hidden_dim, int)
        or projection_hidden_dim <= 0
    ):
        raise ValueError(
            "train.self_flow.projection_hidden_dim must be an integer > 0"
        )

    return {
        "enabled": enabled,
        "mask_ratio": mask_ratio,
        "student_layer_ratio": student_layer_ratio,
        "teacher_layer_ratio": teacher_layer_ratio,
        "student_layer": explicit_layers["student_layer"],
        "teacher_layer": explicit_layers["teacher_layer"],
        "projection": projection,
        "projection_hidden_dim": projection_hidden_dim,
    }


def resolve_self_flow_layers(config: dict, depth: int):
    def resolve(name: str, ratio_name: str):
        explicit = config[name]
        if explicit is not None:
            layer = explicit
        else:
            layer = round(config[ratio_name] * depth)
        if not 1 <= layer <= depth:
            raise ValueError(
                f"train.self_flow.{name} resolves to {layer}, but model depth is {depth}"
            )
        return layer

    student = resolve("student_layer", "student_layer_ratio")
    teacher = resolve("teacher_layer", "teacher_layer_ratio")
    if student >= teacher:
        raise ValueError(
            "Self-Flow requires student_layer < teacher_layer; "
            f"resolved to {student} and {teacher}"
        )
    return student, teacher


def training_recipe(config: dict):
    vae_latent_strategy = str(config.get("vae_latent_strategy", "mode"))
    if vae_latent_strategy not in VAE_LATENT_STRATEGIES:
        choices = ", ".join(sorted(VAE_LATENT_STRATEGIES))
        raise ValueError(
            f"train.vae_latent_strategy must be one of: {choices}"
        )

    raw_weights = config.get(
        "branch_sampling_weights",
        {"edit": 1.0, "instruction_dropout": 0.0, "noop": 0.0},
    )
    if not isinstance(raw_weights, dict):
        raise TypeError("train.branch_sampling_weights must be a mapping")
    unknown_branches = set(raw_weights) - set(TRAINING_BRANCHES)
    if unknown_branches:
        raise ValueError(
            "unknown train.branch_sampling_weights entries: "
            + ", ".join(sorted(unknown_branches))
        )

    branch_weights = {}
    for branch in TRAINING_BRANCHES:
        weight = float(raw_weights.get(branch, 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"train.branch_sampling_weights.{branch} must be finite and >= 0"
            )
        branch_weights[branch] = weight
    if sum(branch_weights.values()) <= 0.0:
        raise ValueError("at least one training branch sampling weight must be > 0")

    raw_noop_prompts = config.get("noop_prompts", [])
    if not isinstance(raw_noop_prompts, list):
        raise TypeError("train.noop_prompts must be a list")
    noop_prompts = []
    for index, prompt in enumerate(raw_noop_prompts):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"train.noop_prompts[{index}] must be a non-empty string"
            )
        noop_prompts.append(prompt.strip())
    if branch_weights["noop"] > 0.0 and not noop_prompts:
        raise ValueError(
            "train.noop_prompts must not be empty when the noop branch weight is > 0"
        )

    return vae_latent_strategy, branch_weights, noop_prompts


def choose_training_branch(sample: dict, branch_weights: dict[str, float]):
    eligible_branches = ["edit", "instruction_dropout"]
    if sample["primary_source_index"] is not None:
        eligible_branches.append("noop")
    eligible_weights = [branch_weights[branch] for branch in eligible_branches]
    total_weight = sum(eligible_weights)
    if total_weight <= 0.0:
        raise ValueError(
            f"sample {sample['id']!r} has no eligible training branch with positive weight"
        )
    return random.choices(eligible_branches, weights=eligible_weights, k=1)[0]


def loss_recipe(config: dict):
    raw_loss = config.get("loss", {})
    if not isinstance(raw_loss, dict):
        raise TypeError("train.loss must be a mapping")
    allowed_entries = {
        "weights",
        "pixel_type",
        "huber_delta",
        "charbonnier_epsilon",
        "gradient_checkpointing",
        "perceptual_dtype",
        "perceptual_backbones",
    }
    unknown_entries = set(raw_loss) - allowed_entries
    if unknown_entries:
        raise ValueError(
            "unknown train.loss entries: " + ", ".join(sorted(unknown_entries))
        )

    raw_weights = raw_loss.get(
        "weights",
        {"fm": 1.0, "perceptual": 0.0, "pixel": 0.0, "self_flow": 0.0},
    )
    if not isinstance(raw_weights, dict):
        raise TypeError("train.loss.weights must be a mapping")
    unknown_weights = set(raw_weights) - set(LOSS_NAMES)
    if unknown_weights:
        raise ValueError(
            "unknown train.loss.weights entries: "
            + ", ".join(sorted(unknown_weights))
        )
    loss_weights = {}
    for name in LOSS_NAMES:
        weight = float(raw_weights.get(name, 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"train.loss.weights.{name} must be finite and >= 0"
            )
        loss_weights[name] = weight
    if sum(loss_weights.values()) <= 0.0:
        raise ValueError("at least one train.loss weight must be > 0")

    pixel_type = str(raw_loss.get("pixel_type", "mse")).lower()
    if pixel_type not in PIXEL_LOSS_TYPES:
        choices = ", ".join(sorted(PIXEL_LOSS_TYPES))
        raise ValueError(f"train.loss.pixel_type must be one of: {choices}")
    huber_delta = float(raw_loss.get("huber_delta", 1.0))
    if not math.isfinite(huber_delta) or huber_delta <= 0.0:
        raise ValueError("train.loss.huber_delta must be finite and > 0")
    charbonnier_epsilon = float(raw_loss.get("charbonnier_epsilon", 1e-3))
    if not math.isfinite(charbonnier_epsilon) or charbonnier_epsilon <= 0.0:
        raise ValueError(
            "train.loss.charbonnier_epsilon must be finite and > 0"
        )
    gradient_checkpointing = raw_loss.get("gradient_checkpointing", True)
    if not isinstance(gradient_checkpointing, bool):
        raise TypeError("train.loss.gradient_checkpointing must be a boolean")
    perceptual_dtype = str(raw_loss.get("perceptual_dtype", "bf16")).lower()
    if perceptual_dtype not in {"bf16", "fp32"}:
        raise ValueError(
            "train.loss.perceptual_dtype must be bf16 or fp32"
        )
    perceptual_backbones = raw_loss.get(
        "perceptual_backbones", DEFAULT_PERCEPTUAL_BACKBONES
    )
    if loss_weights["perceptual"] > 0.0:
        if not isinstance(perceptual_backbones, list) or not perceptual_backbones:
            raise ValueError(
                "train.loss.perceptual_backbones must be a non-empty list "
                "when perceptual loss is enabled"
            )

    return {
        "weights": loss_weights,
        "pixel_type": pixel_type,
        "huber_delta": huber_delta,
        "charbonnier_epsilon": charbonnier_epsilon,
        "gradient_checkpointing": gradient_checkpointing,
        "perceptual_dtype": perceptual_dtype,
        "perceptual_backbones": perceptual_backbones,
    }


def pixel_space_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    huber_delta: float,
    charbonnier_epsilon: float,
):
    prediction = prediction.float()
    target = target.float()
    if loss_type == "mse":
        return F.mse_loss(prediction, target)
    if loss_type == "l1":
        return F.l1_loss(prediction, target)
    if loss_type == "huber":
        return F.huber_loss(prediction, target, delta=huber_delta)
    difference = prediction - target
    return torch.sqrt(
        difference.square() + charbonnier_epsilon * charbonnier_epsilon
    ).mean()


def decode_flow_endpoints(
    conditioning,
    predictions: list[torch.Tensor],
    noisy_targets: list[torch.Tensor],
    target_images: list[torch.Tensor],
    timestep_fields: list[torch.Tensor],
    patch: int,
    use_gradient_checkpointing: bool,
):
    predicted_pixels = []
    target_pixels = []
    for prediction, noisy, target_image, timestep_field in zip(
        predictions, noisy_targets, target_images, timestep_fields
    ):
        velocity = velocity_tokens_to_latent(
            prediction,
            noisy.shape[-2],
            noisy.shape[-1],
            patch,
        )
        predicted_clean = (
            noisy.float() - timestep_field.float() * velocity.float()
        )
        if use_gradient_checkpointing:
            decoded_prediction = activation_checkpoint(
                conditioning.decode_latent_to_pixels,
                predicted_clean,
                use_reentrant=False,
            )
        else:
            decoded_prediction = conditioning.decode_latent_to_pixels(
                predicted_clean
            )
        with torch.no_grad():
            decoded_target = target_image.to(
                device=decoded_prediction.device,
                dtype=torch.float32,
                non_blocking=True,
            ).mul(2.0).sub(1.0)
            if decoded_target.shape != decoded_prediction.shape:
                raise RuntimeError(
                    "decoded prediction and pre-encode target must have identical "
                    f"C/H/W, got {tuple(decoded_prediction.shape)} and "
                    f"{tuple(decoded_target.shape)}"
                )
        predicted_pixels.append(decoded_prediction)
        target_pixels.append(decoded_target)
    return predicted_pixels, target_pixels


def dual_timestep_fields(
    clean_targets: list[torch.Tensor],
    timesteps: torch.Tensor,
    alternate_timesteps: torch.Tensor,
    alternate_target_masks: list[torch.Tensor],
    patch: int,
):
    fields = []
    for clean, timestep, alternate_timestep, alternate_mask in zip(
        clean_targets,
        timesteps,
        alternate_timesteps,
        alternate_target_masks,
    ):
        height, width = clean.shape[-2:]
        grid_height, grid_width = height // patch, width // patch
        if alternate_mask.shape != (grid_height * grid_width,):
            raise ValueError("dual-timestep mask does not match target token count")
        token_times = torch.where(
            alternate_mask,
            alternate_timestep,
            timestep,
        ).reshape(grid_height, grid_width)
        fields.append(
            token_times.repeat_interleave(patch, dim=0)
            .repeat_interleave(patch, dim=1)
            .unsqueeze(0)
        )
    return fields


def dual_timestep_flow_loss(
    predictions: list[torch.Tensor],
    clean_targets: list[torch.Tensor],
    noises: list[torch.Tensor],
    timestep_loss_weights: torch.Tensor,
    alternate_timestep_loss_weights: torch.Tensor,
    alternate_target_masks: list[torch.Tensor],
    patch: int,
):
    sample_losses = []
    for (
        prediction,
        clean,
        noise,
        weight,
        alternate_weight,
        alternate_mask,
    ) in zip(
        predictions,
        clean_targets,
        noises,
        timestep_loss_weights,
        alternate_timestep_loss_weights,
        alternate_target_masks,
    ):
        target = target_velocity_tokens(clean, noise, patch).float()
        token_errors = (prediction.float() - target).square().mean(dim=-1)
        token_weights = torch.where(
            alternate_mask,
            alternate_weight,
            weight,
        ).float()
        sample_losses.append((token_errors * token_weights).mean())
    return torch.stack(sample_losses).mean()


def self_flow_representation_loss(
    student_features: list[torch.Tensor],
    teacher_features: list[torch.Tensor],
):
    if len(student_features) != len(teacher_features):
        raise RuntimeError("Self-Flow student/teacher batch sizes differ")
    losses = []
    for student, teacher in zip(student_features, teacher_features):
        if student.shape != teacher.shape:
            raise RuntimeError(
                "Self-Flow student/teacher feature shapes differ: "
                f"{tuple(student.shape)} vs {tuple(teacher.shape)}"
            )
        losses.append(
            1.0
            - F.cosine_similarity(
                student.float(), teacher.float(), dim=-1
            ).mean()
        )
    return torch.stack(losses).mean()


def ema_teacher_features(
    model,
    ema,
    noisy_targets: list[torch.Tensor],
    reference_latents: list[list[torch.Tensor]],
    contexts: list[torch.Tensor],
    timesteps: torch.Tensor,
    representation_layer: int,
):
    """Run the stateless EMA teacher in eval/no-grad mode and return raw features."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, features = functional_call(
                model,
                ema.functional_parameters(model, dtype=torch.bfloat16),
                (),
                {
                    "noisy_targets": noisy_targets,
                    "references": reference_latents,
                    "contexts": contexts,
                    "timesteps": timesteps,
                    "representation_layer": representation_layer,
                    "project_representation": False,
                    "return_velocity": False,
                },
                strict=False,
            )
    finally:
        model.train(was_training)
    return features


def parse_args():
    parser = argparse.ArgumentParser(description="Krea2Edit ragged LoRA trainer")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config")
    mode.add_argument("--convert-adapter", type=Path)
    mode.add_argument("--build-dual-svdquant", type=Path)
    parser.add_argument("--convert-output", type=Path)
    return parser.parse_args()


def make_optimizer(model, config):
    optimizer_name = str(config["optimizer"]).lower()
    named_parameters = list(model.named_parameters())
    parameters = [
        parameter for _, parameter in named_parameters if parameter.requires_grad
    ]
    if optimizer_name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            parameters,
            lr=float(config["lr"]),
            betas=tuple(config["betas"]),
            weight_decay=float(config["weight_decay"]),
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(config["lr"]),
            betas=tuple(config["betas"]),
            weight_decay=float(config["weight_decay"]),
        )
    if optimizer_name in DEEPSPEED_MUON_NAMES:
        from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam

        for name, parameter in named_parameters:
            lowered = name.lower()
            parameter.use_muon = (
                parameter.ndim >= 2
                and "embed" not in lowered
                and "lm_head" not in lowered
            )

        muon_parameters = [
            parameter
            for _, parameter in named_parameters
            if parameter.requires_grad and parameter.use_muon
        ]
        adam_parameters = [
            parameter
            for _, parameter in named_parameters
            if parameter.requires_grad and not parameter.use_muon
        ]
        parameter_groups = []
        if muon_parameters:
            parameter_groups.append(
                {
                    "name": "muon-params",
                    "params": muon_parameters,
                    "use_muon": True,
                    "lr": float(config["muon_lr"]),
                    "momentum": float(config["muon_momentum"]),
                    "weight_decay": float(config["muon_weight_decay"]),
                    "ns_method": config["muon_ns_method"],
                }
            )
        if adam_parameters:
            parameter_groups.append(
                {
                    "name": "adam-params",
                    "params": adam_parameters,
                    "use_muon": False,
                    "lr": float(config["adam_lr"]),
                    "betas": tuple(config["adam_betas"]),
                    "eps": float(config["adam_eps"]),
                    "weight_decay": float(config["adam_weight_decay"]),
                }
            )
        return MuonWithAuxAdam(
            parameter_groups,
            adam_optimizer=torch.optim.AdamW,
            adam_optimizer_kwargs={},
            adam_w_mode=True,
        )
    raise ValueError(f"Unknown optimizer: {config['optimizer']}")


def verify_deepspeed_muon_optimizer(optimizer):
    from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam

    chain = []
    current = optimizer
    while current is not None:
        chain.append(type(current).__name__)
        if isinstance(current, MuonWithAuxAdam):
            return " -> ".join(chain)
        current = next(
            (
                getattr(current, name)
                for name in ("optimizer", "_optimizer", "inner_optimizer")
                if getattr(current, name, None) is not None
            ),
            None,
        )
    raise RuntimeError(
        "DeepSpeed Muon was requested, but MuonWithAuxAdam is absent from the "
        f"prepared optimizer chain: {' -> '.join(chain)}"
    )


def save_checkpoint(accelerator, model, ema, output_dir: Path, step: int):
    checkpoint = output_dir / f"checkpoint-{step:08d}"
    accelerator.save_state(checkpoint)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        adapter_dir = checkpoint / "adapter"
        unwrapped_model.dit.save_pretrained(
            adapter_dir, safe_serialization=True
        )
        export_comfyui_lora(
            adapter_dir / "adapter_model.safetensors",
            checkpoint / "krea2edit_comfyui.safetensors",
        )
        if ema is not None:
            ema.save(checkpoint / "ema_state.pt")
            with ema.average_parameters(unwrapped_model):
                ema_adapter_dir = checkpoint / "adapter_ema"
                unwrapped_model.dit.save_pretrained(
                    ema_adapter_dir, safe_serialization=True
                )
                export_comfyui_lora(
                    ema_adapter_dir / "adapter_model.safetensors",
                    checkpoint / "krea2edit_comfyui_ema.safetensors",
                )
    accelerator.wait_for_everyone()


def run_previews(
    accelerator,
    model,
    conditioning,
    dataset,
    config,
    output_dir: Path,
    step: int,
    logging_backend,
    ema,
):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        if ema is None:
            result = generate_previews(
                unwrapped_model,
                conditioning,
                dataset,
                config,
                output_dir,
                step,
            )
        else:
            with ema.average_parameters(unwrapped_model):
                result = generate_previews(
                    unwrapped_model,
                    conditioning,
                    dataset,
                    config,
                    output_dir,
                    step,
                )
        accelerator.print(
            f"step={step} sample_grid={result['rows']}x{result['columns']} "
            f"samples={len(result['sample_ids'])} path={result['path']}"
        )
        if logging_backend == "wandb":
            import wandb

            accelerator.log(
                {
                    "samples/previews": wandb.Image(
                        str(result["path"]),
                        caption=(
                            f"{result['rows']}x{result['columns']} grid, "
                            f"{len(result['sample_ids'])} samples"
                        ),
                    )
                },
                step=step,
            )
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    if args.convert_adapter:
        output = args.convert_output or args.convert_adapter.with_name(
            "krea2edit_comfyui.safetensors"
        )
        export_comfyui_lora(
            args.convert_adapter,
            output,
        )
        print(output)
        return
    if args.build_dual_svdquant:
        output = args.convert_output or args.build_dual_svdquant.with_name(
            f"{args.build_dual_svdquant.stem}-dual.safetensors"
        )
        convert_forward_svdquant_checkpoint(args.build_dual_svdquant, output)
        print(output)
        return

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_config = config["train"]
    vae_latent_strategy, branch_sampling_weights, noop_prompts = training_recipe(
        train_config
    )
    configured_loss = loss_recipe(train_config)
    configured_ema = ema_recipe(train_config)
    configured_self_flow = self_flow_recipe(train_config)
    configured_loss_weights = configured_loss["weights"]
    self_flow_enabled = configured_self_flow["enabled"]
    self_flow_teacher_enabled = (
        self_flow_enabled and configured_loss_weights["self_flow"] > 0.0
    )
    if configured_loss_weights["self_flow"] > 0.0 and not self_flow_enabled:
        raise ValueError(
            "train.loss.weights.self_flow > 0 requires train.self_flow.enabled: true"
        )
    if self_flow_teacher_enabled and not configured_ema["enabled"]:
        raise ValueError("Self-Flow representation loss requires train.ema.enabled: true")
    if self_flow_teacher_enabled and configured_ema["device"] != "accelerator":
        raise ValueError(
            "Self-Flow teacher requires train.ema.device: accelerator"
        )
    if self_flow_teacher_enabled and config["model"]["dtype"] != "bf16":
        raise ValueError("Self-Flow EMA teacher requires model.dtype: bf16")
    decoded_loss_enabled = (
        configured_loss_weights["perceptual"] > 0.0
        or configured_loss_weights["pixel"] > 0.0
    )
    sample_vae_posterior = vae_latent_strategy == "independent_sample"
    optimizer_name = str(train_config["optimizer"]).lower()
    use_deepspeed_muon = optimizer_name in DEEPSPEED_MUON_NAMES
    if use_deepspeed_muon and not config["distributed"]["deepspeed_zero2"]:
        raise ValueError("deepspeed_muon requires distributed.deepspeed_zero2: true")
    deepspeed = None
    if config["distributed"]["deepspeed_zero2"]:
        deepspeed = DeepSpeedPlugin(hf_ds_config={
            "train_micro_batch_size_per_gpu": int(train_config["batch_size"]),
            "gradient_accumulation_steps": int(train_config["gradient_accumulation"]),
            "gradient_clipping": float(train_config["max_grad_norm"]),
            "bf16": {"enabled": config["model"]["dtype"] == "bf16"},
            "fp16": {"enabled": config["model"]["dtype"] == "fp16"},
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_scatter": not use_deepspeed_muon,
                "offload_optimizer": {"device": "none"},
                "offload_param": {"device": "none"},
            },
        })
    mixed_precision = None if config["model"]["dtype"] == "fp32" else config["model"]["dtype"]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_config["gradient_accumulation"]),
        mixed_precision=mixed_precision,
        deepspeed_plugin=deepspeed,
        log_with=config["logging"].get("backend"),
        project_dir=config["output_dir"],
    )
    seed = int(config["seed"])
    set_seed(seed)
    random.seed(seed + accelerator.process_index)

    output_dir = Path(config["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    data_config = config["data"]
    dataset = EditManifestDataset(
        data_config["manifest"],
        int(data_config["min_image_pixels"]),
        int(data_config["max_image_pixels"]),
        int(data_config["alpha_transparency_threshold"]),
        show_progress=accelerator.is_main_process,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        collate_fn=ragged_collate,
    )

    dtype = torch_dtype(config["model"]["dtype"])
    perceptual_criterion = None
    if configured_loss_weights["perceptual"] > 0.0:
        perceptual_criterion = PerceptualLossEnsemble(
            configured_loss["perceptual_backbones"],
            gradient_checkpointing=configured_loss[
                "gradient_checkpointing"
            ],
        ).to(
            device=accelerator.device,
            dtype=torch_dtype(configured_loss["perceptual_dtype"]),
        )
    dit = load_dit(config["model"]["dit"], dtype)
    dit_quantization = config["model"]["quantization"]["dit"]
    dit_quantization_backend = dit_quantization["backend"]
    dual_svdquant_count = 0
    if dit_quantization_backend == "svdquant_dual":
        dit = add_lora(dit, config["lora"])
        svdquant_checkpoint = checkpoint_path(
            dit_quantization["name_or_path"], dit_quantization.get("filename")
        )
        dual_svdquant_count = load_dual_svdquant_linears(
            dit.base_model.model, svdquant_checkpoint
        )
    else:
        quantize_component(dit, dit_quantization)
        dit = add_lora(dit, config["lora"])
    if train_config["gradient_checkpointing"]:
        dit.base_model.model.enable_gradient_checkpointing()
    model = RaggedEditModel(
        dit,
        representation_projection=(
            configured_self_flow["projection"]
            if self_flow_teacher_enabled
            else "none"
        ),
        projection_hidden_dim=configured_self_flow["projection_hidden_dim"],
    )
    self_flow_student_layer = None
    self_flow_teacher_layer = None
    if self_flow_teacher_enabled:
        self_flow_student_layer, self_flow_teacher_layer = resolve_self_flow_layers(
            configured_self_flow, model.depth
        )

    optimizer = make_optimizer(model, train_config)
    optimizer_group_names = [group.get("name") for group in optimizer.param_groups]
    muon_trainable_elements = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and getattr(parameter, "use_muon", False)
    )
    adam_trainable_elements = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and not getattr(parameter, "use_muon", False)
    )
    scheduler = get_scheduler(
        train_config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=int(train_config["warmup_steps"]),
        num_training_steps=int(train_config["steps"]),
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    muon_optimizer_chain = (
        verify_deepspeed_muon_optimizer(optimizer) if use_deepspeed_muon else None
    )
    dataloader = accelerator.prepare_data_loader(dataloader, device_placement=False)

    ema = None
    if configured_ema["enabled"] and (
        accelerator.is_main_process or self_flow_teacher_enabled
    ):
        ema_device = (
            torch.device("cpu")
            if configured_ema["device"] == "cpu"
            else accelerator.device
        )
        ema = TrainableParameterEMA(
            accelerator.unwrap_model(model),
            decay=configured_ema["decay"],
            warmup_steps=configured_ema["warmup_steps"],
            update_every=configured_ema["update_every"],
            device=ema_device,
            include_prefixes=("dit.",),
        )

    conditioning = ConditioningModels(config["model"], dtype, accelerator.device)
    accelerator.init_trackers(config["project_name"], config=config)
    sample_config = config.get("sample", {})
    sampling_enabled = bool(sample_config.get("enabled", False))
    sample_dataset = None
    if sampling_enabled and accelerator.is_main_process:
        preview_source = EditManifestDataset(
            sample_config["manifest"],
            int(data_config["min_image_pixels"]),
            int(data_config["max_image_pixels"]),
            int(data_config["alpha_transparency_threshold"]),
            show_progress=True,
        )
        sample_dataset = PreviewDatasetView(
            preview_source,
            [sample["id"] for sample in sample_config["samples"]],
        )
    if accelerator.is_main_process:
        te_quantization = config["model"]["quantization"]["text_encoder"]
        accelerator.print(
            f"samples={len(dataset)} batch={train_config['batch_size']} "
            f"optimizer={optimizer_name} "
            f"dit_quant_backend={dit_quantization_backend} "
            f"dit_quant_weights={dit_quantization.get('weights', 'n/a')} "
            f"dual_svdquant_linears={dual_svdquant_count} "
            f"te_quant_backend={te_quantization['backend']} "
            f"te_quant_weights={te_quantization.get('weights', 'n/a')}"
        )
        if sampling_enabled:
            accelerator.print(
                f"sample_every={sample_config['every']} "
                f"sample_count={len(sample_config['samples'])}"
            )
        if muon_optimizer_chain:
            accelerator.print(
                f"optimizer_chain={muon_optimizer_chain} "
                f"muon_trainable_elements={muon_trainable_elements} "
                f"adam_trainable_elements={adam_trainable_elements} "
                "reduce_scatter=false"
            )
        if ema is not None:
            accelerator.print(
                f"ema_decay={ema.decay} ema_warmup_steps="
                f"{ema.warmup_steps} ema_update_every={ema.update_every} "
                f"ema_device={ema.device} ema_trainable_elements={ema.parameter_count}"
            )
        if self_flow_enabled:
            self_flow_line = (
                f"self_flow_mask_ratio={configured_self_flow['mask_ratio']}"
            )
            if self_flow_teacher_enabled:
                self_flow_line += (
                    f" self_flow_student_layer={self_flow_student_layer}"
                    f" self_flow_teacher_layer={self_flow_teacher_layer}"
                    f" self_flow_projection={configured_self_flow['projection']}"
                )
            accelerator.print(self_flow_line)

    global_step = 0
    resume_from = train_config.get("resume_from")
    if resume_from:
        accelerator.load_state(resume_from)
        global_step = int(Path(resume_from).name.rsplit("-", 1)[-1])
        if ema is not None:
            ema_path = Path(resume_from) / "ema_state.pt"
            if ema_path.is_file():
                ema.load(ema_path, accelerator.unwrap_model(model))
            else:
                ema.reset(accelerator.unwrap_model(model))
                accelerator.print(
                    "EMA state is absent from the resumed checkpoint; initialized "
                    "EMA from the resumed online LoRA weights."
                )

    model.train()
    accumulated_loss = torch.zeros((), device=accelerator.device)
    active_loss_names = tuple(
        name
        for name in LOSS_NAMES
        if configured_loss_weights[name] > 0.0
    )
    active_loss_indices = {
        name: index for index, name in enumerate(active_loss_names)
    }
    accumulated_component_losses = torch.zeros(
        len(active_loss_names),
        device=accelerator.device,
        dtype=torch.float32,
    )
    accumulated_micro_steps = 0
    while global_step < int(train_config["steps"]):
        data_progress = tqdm(
            dataloader,
            desc="Loading data",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not accelerator.is_main_process,
        )
        for batch in data_progress:
            with accelerator.accumulate(model):
                branches = [
                    choose_training_branch(sample, branch_sampling_weights)
                    for sample in batch
                ]
                prompts = []
                for sample, branch in zip(batch, branches):
                    if branch == "edit":
                        prompts.append(sample["prompt"])
                    elif branch == "instruction_dropout":
                        prompts.append("")
                    else:
                        prompts.append(random.choice(noop_prompts))
                loss_target_images = []
                if decoded_loss_enabled:
                    for sample, branch in zip(batch, branches):
                        if branch == "noop":
                            loss_target_images.append(
                                sample["references"][sample["primary_source_index"]]
                            )
                        else:
                            loss_target_images.append(sample["target"])
                with torch.no_grad():
                    contexts = [
                        conditioning.encode_prompt(prompt, sample["references"])
                        for sample, prompt in zip(batch, prompts)
                    ]
                    reference_latents = [
                        [
                            conditioning.encode_image(
                                image, sample_posterior=sample_vae_posterior
                            )
                            for image in sample["references"]
                        ]
                        for sample in batch
                    ]
                    clean_targets = []
                    for sample, branch, sample_reference_latents in zip(
                        batch, branches, reference_latents
                    ):
                        if branch == "noop":
                            clean_targets.append(
                                sample_reference_latents[
                                    sample["primary_source_index"]
                                ]
                            )
                        else:
                            clean_targets.append(
                                conditioning.encode_image(
                                    sample["target"],
                                    sample_posterior=sample_vae_posterior,
                                )
                            )

                patch = accelerator.unwrap_model(model).patch
                token_counts = [
                    (latent.shape[-2] // patch) * (latent.shape[-1] // patch)
                    for latent in clean_targets
                ]
                timesteps, timestep_loss_weights = sample_timesteps(
                    train_config["timestep_sampling"], len(batch), token_counts, accelerator.device
                )
                timesteps = timesteps.to(dtype)
                alternate_timesteps = None
                alternate_timestep_loss_weights = None
                alternate_target_masks = None
                if self_flow_enabled:
                    (
                        alternate_timesteps,
                        alternate_timestep_loss_weights,
                    ) = sample_timesteps(
                        train_config["timestep_sampling"],
                        len(batch),
                        token_counts,
                        accelerator.device,
                    )
                    alternate_timesteps = alternate_timesteps.to(dtype)
                    alternate_target_masks = [
                        torch.rand(count, device=accelerator.device)
                        < configured_self_flow["mask_ratio"]
                        for count in token_counts
                    ]
                noises = [torch.randn_like(latent) for latent in clean_targets]
                if self_flow_enabled:
                    timestep_fields = dual_timestep_fields(
                        clean_targets,
                        timesteps,
                        alternate_timesteps,
                        alternate_target_masks,
                        patch,
                    )
                else:
                    timestep_fields = list(timesteps)
                noisy_targets = [
                    (1.0 - timestep_field) * clean + timestep_field * noise
                    for clean, noise, timestep_field in zip(
                        clean_targets, noises, timestep_fields
                    )
                ]
                if self_flow_teacher_enabled:
                    predictions, student_features = model(
                        noisy_targets,
                        reference_latents,
                        contexts,
                        timesteps,
                        alternate_timesteps=alternate_timesteps,
                        alternate_target_masks=alternate_target_masks,
                        representation_layer=self_flow_student_layer,
                        project_representation=True,
                        return_velocity=True,
                    )
                else:
                    predictions = model(
                        noisy_targets,
                        reference_latents,
                        contexts,
                        timesteps,
                        alternate_timesteps=alternate_timesteps,
                        alternate_target_masks=alternate_target_masks,
                    )
                    student_features = None
                # Keep the final velocity path attached even for a Self-Flow-only
                # ablation, so distributed wrappers do not see later DiT layers as
                # unused parameters.
                loss = predictions[0].float().sum() * 0.0
                component_losses = {}
                if configured_loss_weights["fm"] > 0.0:
                    if self_flow_enabled:
                        fm_loss = dual_timestep_flow_loss(
                            predictions,
                            clean_targets,
                            noises,
                            timestep_loss_weights,
                            alternate_timestep_loss_weights,
                            alternate_target_masks,
                            patch,
                        )
                    else:
                        fm_losses = [
                            F.mse_loss(
                                prediction.float(),
                                target_velocity_tokens(clean, noise, patch).float(),
                            )
                            for prediction, clean, noise in zip(
                                predictions, clean_targets, noises
                            )
                        ]
                        fm_loss = (
                            torch.stack(fm_losses) * timestep_loss_weights
                        ).mean()
                    component_losses["fm"] = fm_loss
                    loss = loss + configured_loss_weights["fm"] * fm_loss

                if self_flow_teacher_enabled:
                    teacher_timesteps = torch.minimum(
                        timesteps, alternate_timesteps
                    )
                    teacher_noisy_targets = [
                        (1.0 - timestep) * clean + timestep * noise
                        for clean, noise, timestep in zip(
                            clean_targets, noises, teacher_timesteps
                        )
                    ]
                    teacher_features = ema_teacher_features(
                        accelerator.unwrap_model(model),
                        ema,
                        teacher_noisy_targets,
                        reference_latents,
                        contexts,
                        teacher_timesteps,
                        self_flow_teacher_layer,
                    )
                    self_flow_value = self_flow_representation_loss(
                        student_features, teacher_features
                    )
                    component_losses["self_flow"] = self_flow_value
                    loss = (
                        loss
                        + configured_loss_weights["self_flow"]
                        * self_flow_value
                    )

                if decoded_loss_enabled:
                    decoded_predictions, pixel_targets = decode_flow_endpoints(
                        conditioning=conditioning,
                        predictions=predictions,
                        noisy_targets=noisy_targets,
                        target_images=loss_target_images,
                        timestep_fields=timestep_fields,
                        patch=patch,
                        use_gradient_checkpointing=configured_loss[
                            "gradient_checkpointing"
                        ],
                    )
                    if configured_loss_weights["pixel"] > 0.0:
                        pixel_losses = [
                            pixel_space_loss(
                                prediction,
                                target,
                                configured_loss["pixel_type"],
                                configured_loss["huber_delta"],
                                configured_loss["charbonnier_epsilon"],
                            )
                            for prediction, target in zip(
                                decoded_predictions, pixel_targets
                            )
                        ]
                        pixel_value = torch.stack(pixel_losses).mean()
                        component_losses["pixel"] = pixel_value
                        loss = (
                            loss
                            + configured_loss_weights["pixel"] * pixel_value
                        )
                    if configured_loss_weights["perceptual"] > 0.0:
                        perceptual_value = perceptual_criterion(
                            decoded_predictions,
                            pixel_targets,
                        )
                        component_losses["perceptual"] = perceptual_value
                        loss = (
                            loss
                            + configured_loss_weights["perceptual"]
                            * perceptual_value
                        )
                accumulated_loss += loss.detach()
                for name, component_loss in component_losses.items():
                    accumulated_component_losses[
                        active_loss_indices[name]
                    ] += component_loss.detach()
                accumulated_micro_steps += 1

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), float(train_config["max_grad_norm"])
                    )
                    muon_update_norm = None
                    if use_deepspeed_muon:
                        muon_update_norm = getattr(
                            accelerator.deepspeed_engine_wrapped.engine.optimizer,
                            "_muon_update_norm",
                            None,
                        )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if ema is not None and not accelerator.optimizer_step_was_skipped:
                    ema.update(accelerator.unwrap_model(model), global_step)
                mean_loss = accelerator.reduce(
                    accumulated_loss / accumulated_micro_steps, reduction="mean"
                ).item()
                mean_component_values = accelerator.reduce(
                    accumulated_component_losses / accumulated_micro_steps,
                    reduction="mean",
                )
                mean_component_losses = {
                    name: mean_component_values[index].item()
                    for index, name in enumerate(active_loss_names)
                }
                accumulated_loss.zero_()
                accumulated_component_losses.zero_()
                accumulated_micro_steps = 0
                learning_rates = scheduler.get_last_lr()
                metrics = {
                    "train/loss": mean_loss,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": learning_rates[0],
                }
                metrics.update(
                    {
                        f"train/loss_{name}": value
                        for name, value in mean_component_losses.items()
                    }
                )
                if muon_update_norm is not None:
                    metrics["train/muon_update_norm_pre_clip"] = float(muon_update_norm)
                if use_deepspeed_muon:
                    for group_name, learning_rate in zip(
                        optimizer_group_names, learning_rates
                    ):
                        if group_name == "muon-params":
                            metrics["train/lr_muon"] = learning_rate
                        elif group_name == "adam-params":
                            metrics["train/lr_adam"] = learning_rate
                accelerator.log(metrics, step=global_step)
                if global_step % int(train_config["log_every"]) == 0:
                    log_line = (
                        f"step={global_step} loss={mean_loss:.6f} "
                        f"grad_norm={float(grad_norm):.6f}"
                    )
                    for name in active_loss_names:
                        log_line += (
                            f" {name}_loss="
                            f"{mean_component_losses[name]:.6f}"
                        )
                    if muon_update_norm is not None:
                        log_line += f" muon_update_norm_pre_clip={float(muon_update_norm):.6f}"
                    accelerator.print(log_line)
                if global_step % int(train_config["save_every"]) == 0:
                    save_checkpoint(
                        accelerator, model, ema, output_dir, global_step
                    )
                if (
                    sampling_enabled
                    and global_step % int(sample_config["every"]) == 0
                ):
                    run_previews(
                        accelerator,
                        model,
                        conditioning,
                        sample_dataset,
                        sample_config,
                        output_dir,
                        global_step,
                        config["logging"].get("backend"),
                        ema,
                    )
                if global_step >= int(train_config["steps"]):
                    break
        data_progress.close()

    if global_step % int(train_config["save_every"]) != 0:
        save_checkpoint(accelerator, model, ema, output_dir, global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
