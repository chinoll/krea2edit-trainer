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

[grounded text | ref_1 | ... | ref_N | noisy target]
        │
        └─ SingleStreamDiT
             ├─ text:   position=(0,0,0), sampled t
             ├─ ref_i:  frame=i, independent (h,w), t=0 clean
             └─ target: frame=0, independent (H,W), sampled t
```

参考图和目标图各自从 `(0,0)` 开始生成二维 RoPE 网格，没有 crop、padding 或目标图坐标偏移。每张图先保持比例执行单图像素上限，再轻微缩放到 16px 网格；因此原始输入 H/W 可以任意，目标和参考图也不需要相同比例或分辨率。

## Ragged batch 与注意力后端

DataLoader 不 stack 图像。一个 batch 内的 target H/W、参考图数量、参考图 H/W 和文本长度均可不同，也不需要 buckets。VAE 逐图编码后，每个样本形成一条独立序列，再用 `cu_seqlens` pack：

```text
sample 0: [text_0 | refs_0 | target_0]
sample 1: [text_1 | refs_1 | target_1]
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
{"id":"edit-0001","target":{"image":"targets/0001.png","caption":"Place the subject on a beach at sunset."},"references":[{"id":"scene","frame":1,"image":"refs/0001_scene.jpg"},{"id":"subject","frame":2,"image":"refs/0001_subject.png"}]}
{"id":"edit-0002","target":{"image":"targets/0002.jpg","caption":"Make the jacket red."},"references":[{"id":"subject","frame":2,"image":"refs/0002_subject.jpg"}]}
```

路径相对于 manifest 所在目录解析，也可以写绝对路径。字段含义：

- `id`：样本 ID。
- `target.image`：监督目标图。
- `target.caption` 或 `target.prompt`：编辑指令。
- `references`：一张或多张参考图；`frame` 是稳定的正整数 RoPE frame ID。

参考图按 `frame` 排序。某种角色缺失时不需要重新编号，例如只有 subject 时仍可使用 `frame: 2`。
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

两项均支持 `none`、`int4`、`int8`、`float8`。例如只量化 VLM：

```yaml
model:
  quantization:
    dit: none
    text_encoder: int8
```

DIT 基座先由 Optimum Quanto 量化并冻结，然后 PEFT 在量化 Linear 上挂载 LoRA；TE 只参与无梯度的多模态编码。VAE 保持训练 dtype，不跟随这两个量化开关。

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
- `logit_normal`、`mode`：Diffusers density schemes。
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

## Checkpoint

每次保存会写入：

```text
output/krea2edit/checkpoint-00000250/
├─ adapter/                 # PEFT LoRA adapter_model.safetensors + config
├─ Accelerate/DeepSpeed model、optimizer 与 scheduler state
└─ random_states_*.pkl
```

从断点继续时设置：

```yaml
train:
  resume_from: output/krea2edit/checkpoint-00000250
```

## 代码边界

- 独立入口：[train.py](train.py)
- ragged manifest 数据层：[krea2edit/data.py](krea2edit/data.py)
- 权重、VLM/VAE、Quanto、PEFT 与 packed model：[krea2edit/modeling.py](krea2edit/modeling.py)
- timestep 策略：[krea2edit/timesteps.py](krea2edit/timesteps.py)
- DiT 与 FA2/FA4 实现：[krea2edit/mmdit.py](krea2edit/mmdit.py)
