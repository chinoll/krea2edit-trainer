from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn


class TrainableParameterEMA:
    """Device-resident FP32 EMA for only the parameters being trained."""

    _STATE_VERSION = 2

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float,
        warmup_steps: int,
        update_every: int,
        device: torch.device,
        include_prefixes: tuple[str, ...] | None = None,
    ):
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.update_every = int(update_every)
        self.device = torch.device(device)
        self.include_prefixes = include_prefixes
        self.parameter_names = tuple(
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                include_prefixes is None
                or any(name.startswith(prefix) for prefix in include_prefixes)
            )
        )
        if not self.parameter_names:
            raise ValueError("EMA requires at least one trainable parameter")
        self.shadow_parameters: dict[str, torch.Tensor] = {}
        self.num_updates = 0
        self.last_update_step = 0
        self.reset(model)

    def _named_trainable_parameters(
        self, model: nn.Module
    ) -> list[tuple[str, nn.Parameter]]:
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                self.include_prefixes is None
                or any(name.startswith(prefix) for prefix in self.include_prefixes)
            )
        }
        current_names = tuple(parameters)
        if current_names != self.parameter_names:
            expected = set(self.parameter_names)
            current = set(current_names)
            missing = sorted(expected - current)
            unexpected = sorted(current - expected)
            raise RuntimeError(
                "EMA trainable parameter set changed; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return [(name, parameters[name]) for name in self.parameter_names]

    def _copy_parameter(self, parameter: nn.Parameter) -> torch.Tensor:
        return parameter.detach().to(
            device=self.device,
            dtype=torch.float32,
            copy=True,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.shadow_parameters.values())

    def reset(self, model: nn.Module):
        self.shadow_parameters = {
            name: self._copy_parameter(parameter)
            for name, parameter in self._named_trainable_parameters(model)
        }
        self.num_updates = 0
        self.last_update_step = 0

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> bool:
        step = int(step)
        if step <= self.last_update_step:
            raise ValueError(
                f"EMA step must increase, got {step} after {self.last_update_step}"
            )
        in_warmup = step <= self.warmup_steps
        if not in_warmup and (step - self.warmup_steps) % self.update_every != 0:
            return False

        parameters = self._named_trainable_parameters(model)
        effective_decay = 0.0 if in_warmup else self.decay
        one_minus_decay = 1.0 - effective_decay
        for name, parameter in parameters:
            shadow = self.shadow_parameters[name]
            value = parameter.detach().to(
                device=self.device,
                dtype=torch.float32,
            )
            shadow.mul_(effective_decay).add_(value, alpha=one_minus_decay)

        self.num_updates += 1
        self.last_update_step = step
        return True

    def state_dict(self) -> dict:
        return {
            "version": self._STATE_VERSION,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "last_update_step": self.last_update_step,
            "parameter_names": list(self.parameter_names),
            "include_prefixes": (
                list(self.include_prefixes)
                if self.include_prefixes is not None
                else None
            ),
            "shadow_parameters": {
                name: parameter.detach().cpu()
                for name, parameter in self.shadow_parameters.items()
            },
        }

    def load_state_dict(self, state: dict, model: nn.Module):
        if int(state.get("version", -1)) != self._STATE_VERSION:
            raise ValueError(
                f"unsupported EMA state version: {state.get('version')!r}"
            )
        checkpoint_recipe = (
            float(state["decay"]),
            int(state["warmup_steps"]),
            int(state["update_every"]),
        )
        current_recipe = (
            self.decay,
            self.warmup_steps,
            self.update_every,
        )
        if checkpoint_recipe != current_recipe:
            raise ValueError(
                "EMA checkpoint recipe does not match the current config: "
                f"checkpoint={checkpoint_recipe}, current={current_recipe}"
            )

        checkpoint_names = tuple(state["parameter_names"])
        if checkpoint_names != self.parameter_names:
            raise ValueError("EMA checkpoint parameter names do not match the model")
        checkpoint_prefixes = state.get("include_prefixes")
        checkpoint_prefixes = (
            tuple(checkpoint_prefixes) if checkpoint_prefixes is not None else None
        )
        if checkpoint_prefixes != self.include_prefixes:
            raise ValueError("EMA checkpoint parameter filter does not match the model")
        model_parameters = dict(self._named_trainable_parameters(model))
        checkpoint_parameters = state["shadow_parameters"]
        if set(checkpoint_parameters) != set(self.parameter_names):
            raise ValueError("EMA checkpoint parameter set is incomplete")

        loaded_parameters = {}
        for name in self.parameter_names:
            value = checkpoint_parameters[name]
            if tuple(value.shape) != tuple(model_parameters[name].shape):
                raise ValueError(
                    f"EMA shape mismatch for {name}: checkpoint={tuple(value.shape)}, "
                    f"model={tuple(model_parameters[name].shape)}"
                )
            loaded_parameters[name] = value.to(
                device=self.device,
                dtype=torch.float32,
                copy=True,
            )
        self.shadow_parameters = loaded_parameters
        self.num_updates = int(state["num_updates"])
        self.last_update_step = int(state["last_update_step"])
        if self.num_updates < 0 or self.last_update_step < 0:
            raise ValueError("EMA checkpoint counters must be non-negative")

    def save(self, path: Path):
        torch.save(self.state_dict(), path)

    def load(self, path: Path, model: nn.Module):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, model)

    def functional_parameters(
        self,
        model: nn.Module,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, torch.Tensor]:
        """Return BF16 (by default) EMA tensors for a stateless teacher forward."""
        parameters = dict(self._named_trainable_parameters(model))
        mapping = {}
        for name in self.parameter_names:
            shadow = self.shadow_parameters[name]
            parameter = parameters[name]
            if shadow.device != parameter.device:
                raise RuntimeError(
                    "EMA teacher forward requires train.ema.device=accelerator; "
                    f"{name} is on {parameter.device}, but its EMA is on {shadow.device}"
                )
            mapping[name] = shadow.to(dtype=dtype)
        return mapping

    @contextmanager
    def average_parameters(self, model: nn.Module):
        if self.num_updates == 0:
            yield
            return

        parameters = self._named_trainable_parameters(model)
        backups = {name: parameter.detach().clone() for name, parameter in parameters}
        try:
            with torch.no_grad():
                for name, parameter in parameters:
                    parameter.copy_(
                        self.shadow_parameters[name].to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                    )
            yield
        finally:
            with torch.no_grad():
                for name, parameter in parameters:
                    parameter.copy_(backups[name])
