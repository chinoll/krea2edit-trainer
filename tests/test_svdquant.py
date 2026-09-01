import unittest

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from krea2edit.svdquant import (
    DualSVDQuantLinear,
    _pack_int4_weight,
    _pack_lowrank,
    _pack_scale,
    _pack_vector,
    _unpack_int4_weight,
    _unpack_lowrank,
    _unpack_scale,
    _unpack_vector,
    _set_base_layer,
    build_dual_svdquant_state,
)


class SVDQuantTests(unittest.TestCase):
    def test_nunchaku_layout_roundtrips(self):
        generator = torch.Generator().manual_seed(7)
        qweight = torch.randint(-8, 8, (256, 384), generator=generator, dtype=torch.int8)
        scales = torch.randn(256, 6, generator=generator)
        vector = torch.randn(384, generator=generator)
        down = torch.randn(32, 384, generator=generator)
        up = torch.randn(256, 32, generator=generator)

        self.assertTrue(torch.equal(qweight, _unpack_int4_weight(_pack_int4_weight(qweight))))
        self.assertTrue(torch.equal(scales, _unpack_scale(_pack_scale(scales))))
        self.assertTrue(torch.equal(vector, _unpack_vector(_pack_vector(vector))))
        self.assertTrue(
            torch.equal(down, _unpack_lowrank(_pack_lowrank(down, down=True), down=True))
        )
        self.assertTrue(
            torch.equal(up, _unpack_lowrank(_pack_lowrank(up, down=False), down=False))
        )

    def test_dual_checkpoint_transposes_main_and_lowrank_weights(self):
        generator = torch.Generator().manual_seed(11)
        out_features, in_features, rank = 256, 384, 32
        qweight = torch.randint(
            -8, 8, (out_features, in_features), generator=generator, dtype=torch.int8
        )
        scales = torch.rand(out_features, in_features // 64, generator=generator) + 0.05
        smooth = torch.rand(in_features, generator=generator) + 0.5
        down = torch.randn(rank, in_features, generator=generator)
        up = torch.randn(out_features, rank, generator=generator)
        state = {
            "block.qweight": _pack_int4_weight(qweight),
            "block.wscales": _pack_scale(scales),
            "block.smooth_factor": _pack_vector(smooth),
            "block.smooth_factor_orig": _pack_vector(smooth),
            "block.proj_down": _pack_lowrank(down, down=True),
            "block.proj_up": _pack_lowrank(up, down=False),
        }

        dual = build_dual_svdquant_state(state)
        backward_prefix = "block.backward_linear"
        backward_qweight = _unpack_int4_weight(
            dual[f"{backward_prefix}.qweight"]
        ).float()
        backward_scales = _unpack_scale(dual[f"{backward_prefix}.wscales"]).float()
        backward_main = backward_qweight * backward_scales.repeat_interleave(64, dim=1)
        expected_main = (
            qweight.float() * scales.repeat_interleave(64, dim=1) / smooth.unsqueeze(0)
        ).t()
        error = (backward_main - expected_main).abs().view(in_features, -1, 64).amax(-1)
        self.assertTrue(torch.all(error <= backward_scales / 2 + 1e-5))

        backward_down = _unpack_lowrank(
            dual[f"{backward_prefix}.proj_down"], down=True
        )
        backward_up = _unpack_lowrank(
            dual[f"{backward_prefix}.proj_up"], down=False
        )
        self.assertTrue(torch.equal(backward_down, up.t()))
        self.assertTrue(torch.equal(backward_up, down.t()))

    def test_backward_runs_the_transpose_linear(self):
        generator = torch.Generator().manual_seed(19)
        weight = torch.randn(5, 3, generator=generator)
        forward_linear = nn.Linear(3, 5, bias=False)
        backward_linear = nn.Linear(5, 3, bias=False)
        forward_linear.weight.data.copy_(weight)
        backward_linear.weight.data.copy_(weight.t())
        forward_linear.requires_grad_(False)
        backward_linear.requires_grad_(False)
        linear = DualSVDQuantLinear(forward_linear, backward_linear)

        tensor = torch.randn(2, 4, 3, generator=generator, requires_grad=True)
        grad_output = torch.randn(2, 4, 5, generator=generator)
        output = linear(tensor)
        output.backward(grad_output)

        self.assertTrue(torch.allclose(output, torch.nn.functional.linear(tensor, weight)))
        self.assertTrue(torch.allclose(tensor.grad, grad_output @ weight))

    def test_dual_linear_replaces_peft_base_without_changing_lora(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 6)

            def forward(self, tensor):
                return self.proj(tensor)

        peft_model = get_peft_model(
            ToyModel(),
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=["proj"],
                bias="none",
            ),
        )
        root = peft_model.base_model.model
        original = root.proj.get_base_layer()
        forward_linear = nn.Linear(4, 6)
        backward_linear = nn.Linear(6, 4, bias=False)
        forward_linear.load_state_dict(original.state_dict())
        backward_linear.weight.data.copy_(original.weight.data.t())
        forward_linear.requires_grad_(False)
        backward_linear.requires_grad_(False)
        replacement = DualSVDQuantLinear(forward_linear, backward_linear)

        _set_base_layer(root, "proj", replacement)

        self.assertIs(root.proj.get_base_layer(), replacement)
        self.assertIn("default", root.proj.lora_A)
        tensor = torch.randn(3, 4)
        peft_model(tensor).sum().backward()
        self.assertIsNotNone(root.proj.lora_B["default"].weight.grad)
        self.assertFalse(any(parameter.requires_grad for parameter in replacement.parameters()))


if __name__ == "__main__":
    unittest.main()
