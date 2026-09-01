# Krea2Edit Trainer

这是一个独立的 Krea2Edit 训练项目，也是一套全新的多参考图、变长序列架构。训练入口不依赖 AI-Toolkit。

上游项目：

- 原始训练器：[lbouaraba/krea2edit-trainer](https://github.com/lbouaraba/krea2edit-trainer)
- 训练框架参考：[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- 基础设施：[Diffusers](https://github.com/huggingface/diffusers)、[PEFT](https://github.com/huggingface/peft)、[Accelerate](https://github.com/huggingface/accelerate)、[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)

该架构与上游 target-fitted/crop 方案不同，已有旧架构 LoRA 不能直接视为本架构的继续训练起点。

## 训练数据流

```text
reference images + edit prompt
        │
        ├─ Qwen3-VL 完整编码图像与文本
        │      └─ 删除 <|image_pad|> 对应 hidden states
        │          保留经过图像语义增强的语言 token
        │
        └─ Qwen-Image VAE 对每张参考图独立编码

[grounded text | noisy target | ref_1 | ... | ref_N]
        │
        └─ SingleStreamDiT
             ├─ text:   position=(0,0,0), sampled t
             ├─ target: frame=0, independent (H,W), sampled t
             └─ ref_i:  frame=i, independent (h,w), t=0 clean
```

参考图和目标图各自从 `(0,0)` 开始生成二维 RoPE 网格，没有 crop、padding 或目标图坐标偏移。每张图先保持比例执行单图像素上限，再轻微缩放到 16px 网格；因此原始输入 H/W 可以任意，目标和参考图也不需要相同比例或分辨率。

## Ragged batch 与注意力后端

DataLoader 不 stack 图像。一个 batch 内的 target H/W、参考图数量、参考图 H/W 和文本长度均可不同，也不需要 buckets。VAE 逐图编码后，每个样本形成一条独立序列，再用 `cu_seqlens` pack：

```text
sample 0: [text_0 | target_0 | refs_0]
sample 1: [text_1 | target_1 | refs_1]
...
```

| 情况 | 后端 |
| --- | --- |
| `batch_size: 1` | PyTorch SDPA |
| A100/A800、其他 Ampere/Ada，且 `batch_size > 1` | FlashAttention-2 varlen |
| H100/H200/Blackwell，且 `batch_size > 1` | 优先 FlashAttention-4 varlen，FA2 可作为兼容后端 |

这里没有 padded-SDPA 的伪 varlen 回退。A100 安装 FA2：

```bash
pip install flash-attn --no-build-isolation
```

H100/H200/Blackwell 可安装 FA4：

```bash
pip install flash-attn-4
```

## 安装

Python 3.10+，CUDA 环境中执行：

```bash
git clone https://github.com/chinoll/krea2edit-trainer
cd krea2edit-trainer
pip install -r requirements.txt
```

Krea 2 RAW 为 gated model，需要先在 Hugging Face 接受许可，并设置 `HF_TOKEN`。

## 数据集

训练入口只读取 JSONL manifest。每行是一条独立样本：

```jsonl
{"id":"edit-0001","target":{"image":"targets/0001.png","caption":"Place the subject on a beach at sunset."},"references":[{"id":"scene","image":"refs/0001_scene.jpg"},{"id":"subject","image":"refs/0001_subject.png"}]}
{"id":"edit-0002","target":{"image":"targets/0002.jpg","caption":"Make the jacket red."},"references":[{"id":"subject","image":"refs/0002_subject.jpg"}]}
```

路径相对于 manifest 所在目录解析，也可以写绝对路径。字段含义：

- `id`：样本 ID。
- `target.image`：监督目标图。
- `target.caption` 或 `target.prompt`：编辑指令。
- `references`：一张或多张参考图，数组顺序就是传给 ComfyUI 节点的参考图顺序。

RoPE frame 不写入数据：训练时按数组顺序自动分配为 `1..N`。这与
`comfyui-krea2edit` 的 `source_image`、`source_image_b`、`reference_images`
拼接顺序一致；单参考图始终是 `frame=1`。
完整样例见 [configs/manifest.example.jsonl](configs/manifest.example.jsonl)。

## 单图像素上限与任意分辨率

`data.max_image_pixels` 同时作用于 target 和每一张 reference，但按单张图独立计算，不限制一条样本内所有参考图的像素总和。

当 `H × W` 超过上限时，先按 `sqrt(limit / (H × W))` 等比缩小，再把两个边长轻微缩放到最接近的 16 倍数。对齐后若再次超过上限，则向下对齐。设置为 `0` 可关闭像素上限。

## DIT 与 TE 独立量化

配置中的两个开关互不依赖：

```yaml
model:
  quantization:
    dit: int8
    text_encoder: none
```

两项均支持 `none`、`int4`、`int8`、`float8`。DIT 还支持下面单独说明的实验性
`svdquant_dual`。例如只量化 VLM：

```yaml
model:
  quantization:
    dit: none
    text_encoder: int8
```

DIT 基座先由 Optimum Quanto 量化并冻结，然后 PEFT 在量化 Linear 上挂载 LoRA；TE 只参与无梯度的多模态编码。VAE 保持训练 dtype，不跟随这两个量化开关。

## 实验性双向 SVDQuant

> [!WARNING]
> 这是实验性训练后端，请谨慎使用。它在 backward 中再次执行 W4A4 量化，得到的
> `dX` 是近似梯度，尚未经过与默认 BF16/Quanto 路径同等规模的收敛验证。正式长跑
> 前应先进行短程训练，并对比 loss、gradient norm 和固定 seed 样图；不要直接覆盖
> 已有稳定训练配置或 checkpoint。

`svdquant_dual` 让同一个 DiT Linear 保存两套 Nunchaku INT4 权重：

```text
forward: X ── SVDQuant(W) ──> Y
backward: G ── SVDQuant(W.T) ──> dX
```

前向和输入梯度都调用 Nunchaku 已有的 W4A4 Linear kernel。DiT 基座冻结，
所以不会计算 `dW`；PEFT LoRA 的参数梯度仍然使用 BF16 GEMM。该模式同时支持
普通 3D batch 和 FA2/FA4 使用的 2D packed varlen token。

先按照 [Nunchaku](https://github.com/nunchaku-ai/nunchaku) 官方说明安装与当前
PyTorch、CUDA 和 GPU 架构匹配的 wheel。A100 使用 INT4 checkpoint。然后把已有
forward SVDQuant checkpoint 转成训练使用的双向 checkpoint：

```bash
python train.py \
  --build-dual-svdquant weights/krea2edit-svdquant-forward.safetensors \
  --convert-output weights/krea2edit-svdquant-dual.safetensors
```

输入 checkpoint 中每个 Linear 使用原始 DiT module path，并包含
`qweight`、`wscales`、`smooth`/`smooth_factor`、
`lora_down`/`proj_down`、`lora_up`/`proj_up`。转换器会：

1. 还原 forward 主分支的有效权重；
2. 转置并按 group size 64 重新执行 INT4 RTN；
3. 将固定 SVD 低秩分支交换方向后按 Nunchaku layout 重新 pack；
4. 写出 `forward_linear.*` 和 `backward_linear.*` 两套权重。

训练配置：

```yaml
model:
  quantization:
    dit: svdquant_dual
    text_encoder: int8
  svdquant_dual:
    name_or_path: weights/krea2edit-svdquant-dual.safetensors
    filename: null
```

只有输入、输出通道均为 128 倍数且出现在 checkpoint 中的 Linear 会被替换；首尾
小投影保留 BF16。当前实验后端的两个方向都使用 signed INT4 activation；转置
分支使用 `smooth=1`。
这会把长期基座存储从单份约 4-bit 增加到双份约 8-bit，但 backward 不再临时
生成整层 BF16 权重。若已有使用真实 `G=dL/dY` 校准得到的 backward checkpoint，
也可以直接按相同的 `backward_linear.*` 布局加载，不需要运行上述 RTN 转换。

## Timestep 采样

Krea flow convention 为 `t=0 clean`、`t=1 noise`：

```text
noisy = (1 - t) * clean + t * noise
velocity target = noise - clean
```

`train.timestep_sampling.strategy` 支持：

- `linear`：均匀采样。
- `sigmoid`：AI-Toolkit 常用的 sigmoid-normal 分布。
- `weighted`：AI-Toolkit 的 1000 点均匀 timestep 加原始逐 timestep loss-weight 表；权重表已内置，不依赖 AI-Toolkit。
- `logit_normal`、`mode`：Diffusers density schemes；采样结果从 scheduler index
  fraction 转换为 Krea 的 flow time（`t=1` 为噪声）。
- `cosine_map`：均匀采样 timestep，并使用 Diffusers cosmap loss weighting。
- `lognorm_blend`：75% log-normal 与 25% uniform 混合。
- `shift`：按每个 target 的 DiT token 数独立应用 Krea dynamic shift，插值端点为 `(256, 0.5)` 与 `(6400, 1.15)`。
- `one_step`、`two_step`、`four_step`、`eight_step`：离散 timestep 训练。

无论使用哪种 target timestep，所有 reference latent token 始终在 DiT block 内使用独立的 `t=0` modulation。

## DeepSpeed ZeRO-2 训练

复制并修改 [configs/train.yaml](configs/train.yaml)，至少填写 manifest、输出目录、batch size 和量化组合。单卡运行：

```bash
accelerate launch --num_processes 1 train.py \
  --config configs/train.yaml
```

多卡 ZeRO-2：先把 [configs/accelerate_zero2.yaml](configs/accelerate_zero2.yaml) 中的 `num_processes` 改成 GPU 数量，再运行：

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml \
  train.py --config configs/train.yaml
```

`distributed.deepspeed_zero2: true` 会由训练脚本创建 stage-2 plugin；梯度累积与裁剪直接读取训练 YAML。仓库同时提供原生 [configs/deepspeed_zero2.json](configs/deepspeed_zero2.json)，便于接入已有 DeepSpeed launcher。

## W&B 训练记录

先登录 W&B：

```bash
wandb login
```

示例配置默认启用 W&B：

```yaml
project_name: krea2edit

logging:
  backend: wandb
```

`project_name` 是 W&B project 名称。训练器在每次实际 optimizer update 时记录
梯度累积窗口平均值 `train/loss`、裁剪前 global L2 norm `train/grad_norm` 和
`train/lr`；micro-step 不会产生重复记录。将 `logging.backend` 改为 `null` 可以
关闭在线记录。

## Checkpoint

每次保存会写入：

```text
output/krea2edit/checkpoint-00000250/
├─ adapter/                         # PEFT 格式，用于继续训练或 PEFT 加载
├─ krea2edit_comfyui.safetensors    # ComfyUI 可直接加载的 LoRA
├─ Accelerate/DeepSpeed model、optimizer 与 scheduler state
└─ random_states_*.pkl
```

把 `krea2edit_comfyui.safetensors` 复制到 `ComfyUI/models/loras/`，在 Krea 2
模型和 `Krea2EditModelPatch` 之间使用 `Load LoRA Model Only` 加载。导出器会把
PEFT 的 `base_model.model.` 前缀转换成 ComfyUI 的 `diffusion_model.`，并为每个
LoRA 模块写入 `alpha`，因此
`lora.alpha != lora.rank` 时 ComfyUI 的强度也与训练一致。不要把
`adapter/adapter_model.safetensors` 直接交给 ComfyUI；它保留的是 PEFT 命名。

已有 PEFT adapter 可以直接转换，不会启动训练：

```bash
python train.py \
  --convert-adapter output/krea2edit/checkpoint-00000250/adapter/adapter_model.safetensors \
  --convert-output output/krea2edit/krea2edit_comfyui.safetensors
```

转换时的 `alpha` 直接读取同目录 PEFT `adapter_config.json` 中的
`lora_alpha`。

从断点继续时设置：

```yaml
train:
  resume_from: output/krea2edit/checkpoint-00000250
```

## 代码边界

- 独立入口：[train.py](train.py)
- ragged manifest 数据层：[krea2edit/data.py](krea2edit/data.py)
- 权重、VLM/VAE、Quanto、PEFT 与 packed model：[krea2edit/modeling.py](krea2edit/modeling.py)
- 实验性双向 SVDQuant、转置 checkpoint 与 autograd：[krea2edit/svdquant.py](krea2edit/svdquant.py)
- timestep 策略：[krea2edit/timesteps.py](krea2edit/timesteps.py)
- DiT 与 FA2/FA4 实现：[krea2edit/mmdit.py](krea2edit/mmdit.py)
