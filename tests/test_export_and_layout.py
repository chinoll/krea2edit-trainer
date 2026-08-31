import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from krea2edit.modeling import RaggedEditModel, export_comfyui_lora, latent_tokens


class _RecordingDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(patch=2))
        )
        self.call = None

    def forward(self, **kwargs):
        self.call = kwargs
        return kwargs["img"]


class ExportAndLayoutTests(unittest.TestCase):
    def test_peft_adapter_exports_with_comfyui_keys_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory)
            adapter_file = adapter_dir / "adapter_model.safetensors"
            output_file = adapter_dir / "krea2edit_comfyui.safetensors"
            save_file(
                {
                    "base_model.model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
                    "base_model.model.blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
                },
                str(adapter_file),
            )
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 8}), encoding="utf-8"
            )

            export_comfyui_lora(adapter_file, output_file)
            state = load_file(str(output_file))

            prefix = "diffusion_model.blocks.0.attn.wq"
            self.assertEqual(set(state), {
                f"{prefix}.lora_A.weight",
                f"{prefix}.lora_B.weight",
                f"{prefix}.alpha",
            })
            self.assertEqual(state[f"{prefix}.alpha"].item(), 8.0)

    def test_single_sample_layout_matches_comfyui(self):
        dit = _RecordingDiT()
        model = RaggedEditModel(dit)
        target = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        reference_a = torch.full((1, 4, 4), 100.0)
        reference_b = torch.full((1, 2, 4), 200.0)
        context = torch.zeros(3, 1, 1)

        prediction = model(
            [target],
            [[reference_a, reference_b]],
            [context],
            torch.tensor([0.5]),
        )[0]

        target_tokens, _ = latent_tokens(target, patch=2, frame=0)
        self.assertTrue(torch.equal(prediction, target_tokens))
        self.assertEqual(dit.call["ref_token_count"], 6)
        image_positions = dit.call["pos"][0, context.shape[0]:]
        self.assertEqual(image_positions[:, 0].tolist(), [0.0] * 4 + [1.0] * 4 + [2.0] * 2)


if __name__ == "__main__":
    unittest.main()
