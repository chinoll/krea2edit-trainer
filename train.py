#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from krea2edit.data import EditManifestDataset, ragged_collate
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
from krea2edit.svdquant import (
    convert_forward_svdquant_checkpoint,
    load_dual_svdquant_linears,
)
from krea2edit.sampling import PreviewDatasetView, generate_previews
from krea2edit.timesteps import sample_timesteps


DEEPSPEED_MUON_NAMES = {"deepspeed_muon", "muon"}


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


def save_checkpoint(accelerator, model, output_dir: Path, step: int):
    checkpoint = output_dir / f"checkpoint-{step:08d}"
    accelerator.save_state(checkpoint)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        adapter_dir = checkpoint / "adapter"
        accelerator.unwrap_model(model).dit.save_pretrained(
            adapter_dir, safe_serialization=True
        )
        export_comfyui_lora(
            adapter_dir / "adapter_model.safetensors",
            checkpoint / "krea2edit_comfyui.safetensors",
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
):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        result = generate_previews(
            accelerator.unwrap_model(model),
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
    set_seed(int(config["seed"]))

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
    model = RaggedEditModel(dit)

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

    global_step = 0
    resume_from = train_config.get("resume_from")
    if resume_from:
        accelerator.load_state(resume_from)
        global_step = int(Path(resume_from).name.rsplit("-", 1)[-1])

    model.train()
    accumulated_loss = torch.zeros((), device=accelerator.device)
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
                with torch.no_grad():
                    contexts = [
                        conditioning.encode_prompt(sample["prompt"], sample["references"])
                        for sample in batch
                    ]
                    clean_targets = [conditioning.encode_image(sample["target"]) for sample in batch]
                    reference_latents = [
                        [conditioning.encode_image(image) for image in sample["references"]]
                        for sample in batch
                    ]

                patch = accelerator.unwrap_model(model).patch
                token_counts = [
                    (latent.shape[-2] // patch) * (latent.shape[-1] // patch)
                    for latent in clean_targets
                ]
                timesteps, loss_weights = sample_timesteps(
                    train_config["timestep_sampling"], len(batch), token_counts, accelerator.device
                )
                timesteps = timesteps.to(dtype)
                noises = [torch.randn_like(latent) for latent in clean_targets]
                noisy_targets = [
                    (1.0 - timestep) * clean + timestep * noise
                    for clean, noise, timestep in zip(clean_targets, noises, timesteps)
                ]
                predictions = model(
                    noisy_targets,
                    reference_latents,
                    contexts,
                    timesteps,
                )
                losses = [
                    F.mse_loss(prediction.float(), target_velocity_tokens(clean, noise, patch).float())
                    for prediction, clean, noise in zip(predictions, clean_targets, noises)
                ]
                loss = (torch.stack(losses) * loss_weights).mean()
                accumulated_loss += loss.detach()
                accumulated_micro_steps += 1

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), float(train_config["max_grad_norm"])
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                mean_loss = accelerator.reduce(
                    accumulated_loss / accumulated_micro_steps, reduction="mean"
                ).item()
                accumulated_loss.zero_()
                accumulated_micro_steps = 0
                learning_rates = scheduler.get_last_lr()
                metrics = {
                    "train/loss": mean_loss,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": learning_rates[0],
                }
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
                    accelerator.print(
                        f"step={global_step} loss={mean_loss:.6f} "
                        f"grad_norm={float(grad_norm):.6f}"
                    )
                if global_step % int(train_config["save_every"]) == 0:
                    save_checkpoint(accelerator, model, output_dir, global_step)
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
                    )
                if global_step >= int(train_config["steps"]):
                    break
        data_progress.close()

    if global_step % int(train_config["save_every"]) != 0:
        save_checkpoint(accelerator, model, output_dir, global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
