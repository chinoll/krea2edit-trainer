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

参考图和目标图各自从 `(0,0)` 开始生成二维 RoPE 网格，没有位置编码层面的
padding 或目标图坐标偏移。数据加载阶段让 target 与 `role: source` 的编辑输入
始终使用同一个等比缩放系数；它们原始比例可以不同，最终 H/W 也可以不同。若组内
含 RGBA 图，则会先在归一化画布坐标中做共同的透明内容裁剪，再共同缩放；RGB 图
视为整张画布有效，不会被透明裁剪。其他辅助参考图独立执行透明裁剪与单图像素范围
约束。所有结果最后轻微缩放到 16px 网格，辅助参考图仍可使用任意比例和分辨率。

## Ragged batch 与注意力后端

DataLoader 不 stack 图像。一个 batch 内的 target H/W、参考图数量、参考图 H/W 和文本长度均可不同，也不需要 buckets。VAE 逐图编码后，每个样本形成一条独立序列，再用 `cu_seqlens` pack：

训练时主进程会显示 `Loading data` tqdm 进度条，以 batch 为单位跟踪当前一轮
DataLoader 的图像读取、透明处理与尺寸转换进度；其他分布式 rank 不重复输出。

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

训练入口只读取 JSONL manifest。文件以二进制流逐行读取并由 `orjson` 解析，所有
样本元描述在启动时加载到数据集索引，但不会先把整份 manifest 解码成一个大字符串。
主进程的 `Loading manifest metadata` tqdm 按已完成反序列化的字节推进，因此无需
为了计算总行数额外扫描一次文件，同时仍能显示百分比、速度、ETA 与最终样本数。
每行是一条独立样本：

```jsonl
{"id":"edit-0001","target":{"image":"targets/0001.png","caption":"Place the subject on a beach at sunset."},"references":[{"id":"source","role":"source","image":"refs/0001_source.png"},{"id":"subject","role":"reference","image":"refs/0001_subject.png"}]}
{"id":"edit-0002","target":{"image":"targets/0002.png","caption":"Make the jacket red."},"references":[{"id":"source","role":"source","image":"refs/0002_source.png"}]}
```

路径相对于 manifest 所在目录解析，也可以写绝对路径。字段含义：

- `id`：样本 ID。
- `target.image`：监督目标图。
- `target.caption` 或 `target.prompt`：编辑指令。
- `references`：一张或多张参考图，数组顺序就是传给 ComfyUI 节点的参考图顺序。
- `references[].role: source`：被编辑前的输入图。它与 target 使用同一个等比缩放系数；
  若组内存在 RGBA 图，先共同裁掉透明冗余区域，再缩放。不填充、不做非等比拉伸；原始
  画布比例不同也允许，因此两张图最终分辨率可以不同。缩放后各自轻微对齐到 16px 网格。
- `references[].role: reference`：身份、风格等不要求共享缩放比例的辅助参考，独立处理
  尺寸。为兼容旧 manifest，省略 `role` 时第一张 reference 默认为 `source`，后续
  reference 默认为 `reference`；新数据建议始终显式填写。

RoPE frame 不写入数据：训练时按数组顺序自动分配为 `1..N`。这与
`comfyui-krea2edit` 的 `source_image`、`source_image_b`、`reference_images`
拼接顺序一致；单参考图始终是 `frame=1`。
完整样例见 [configs/manifest.example.jsonl](configs/manifest.example.jsonl)。

## 单图像素范围、透明裁剪与任意分辨率

`data.min_image_pixels` 和 `data.max_image_pixels` 同时作用于 target 与所有
reference，不限制一条样本内所有参考图的像素总和。target 与 `role: source`
reference 作为一个缩放组计算同一个等比系数；其他 reference 按单张图独立计算。
普通 RGB 图像同样受最小像素量约束：当 `H × W` 小于下限时按比例放大，超过上限时
按比例缩小，随后把两个边长轻微缩放到 16 的倍数。两个值分别设为 `0` 可以关闭
对应的下限或上限。

RGBA 图像在缩放前执行以下处理：

1. 将 `alpha <= data.alpha_transparency_threshold` 的像素视为完全透明；alpha 与阈值
   均使用 `0..255`，默认阈值 `8` 用于过滤透明通道的数值抖动。
2. 对 target 与所有 `role: source` 图像分别计算非透明内容包围盒，再映射到归一化
   画布坐标取并集；从 target 的并集框向外扩展后，把同一个相对裁剪框应用到整个组。
   辅助 reference 使用自己的包围盒。组中只要有一张普通 RGB 图，就将它视为整张
   画布均不透明，因此保留完整共享画布。
3. 若 target 的包围盒面积小于 `data.min_image_pixels`，在 target 原图边界内围绕内容
   向外扩展；如果整张原图本身仍小于下限，则保留整张图并在下一步等比放大。
4. 裁剪后的 target 与 `role: source` 始终使用完全相同的等比缩放系数；原始比例可以
   不同，因此对齐 16px 网格后的最终 H/W 也可以不同。辅助 reference 独立缩放。
5. 裁剪前先稳定 alpha；裁剪后将剩余透明和半透明区域合成到纯白背景、删除 alpha
   通道，再执行共享或独立缩放。最终交给模型的图像均为 RGB。

因此最小像素量不仅约束透明裁剪的下限，也约束所有普通图像的最终输入尺寸。
训练集和独立 evaluate 数据集使用相同的数据几何。采样输出始终使用预处理后 GT
target 的 H/W，因此生成图与 GT 完全同尺寸。

## DIT 与 TE 独立量化

配置直接指定量化后端，精度是该后端的参数，不再用 `int8` 之类的精度名称代替
后端。DIT 与 TE 的配置互不依赖：

```yaml
model:
  quantization:
    dit:
      backend: quanto
      weights: qfloat8
    text_encoder:
      backend: none
```

通用后端为 `none` 和 `quanto`。选择 `quanto` 时，`weights` 支持 `qint4`、
`qint8`、`qfloat8`。DIT 还支持下面单独说明的实验性 `svdquant_dual` 后端。
默认 DIT 配置使用 `qfloat8`：冻结基座权重以 FP8 保存，激活不量化并保持
`model.dtype: bf16`，即 W8A16。PEFT LoRA 参数也保持训练 dtype。这个路径以降低
常驻权重显存为目标；A100 没有原生 FP8 Tensor Core，Quanto 会在矩阵乘时转换到
可计算的浮点类型，因此不承诺加速。
例如只量化 VLM：

```yaml
model:
  quantization:
    dit:
      backend: none
    text_encoder:
      backend: quanto
      weights: qint8
```

在这个示例中，DIT 的 `backend: none` 表示 **不量化 DIT**；只有 TE 使用
Quanto `qint8`。

这里的 `backend` 只决定冻结基座采用哪种权重后端，不决定参数是否参与训练：

- DIT 始终冻结基座，只训练 PEFT LoRA。`backend: none` 保留 `model.dtype` 权重；
  `backend: quanto` 先把基座转换成指定的 Quanto `weights`，再挂载 LoRA。
- TE 始终以 eval/no-grad 模式生成多模态条件。`backend` 只改变它的权重表示。
- VAE 始终冻结并保持 `model.dtype`，不受 `model.quantization` 控制。

DIT 的 `svdquant_dual` 是独立的实验后端，加载顺序和梯度近似方式见下一节。

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
    dit:
      backend: svdquant_dual
      name_or_path: weights/krea2edit-svdquant-dual.safetensors
      filename: null
    text_encoder:
      backend: quanto
      weights: qint8
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

### DeepSpeed Muon

DeepSpeed 0.19.6 及以上可使用其原生 ZeRO-2 `MuonWithAuxAdam`：

```yaml
distributed:
  deepspeed_zero2: true

train:
  optimizer: deepspeed_muon
  muon_lr: 2.0e-3
  muon_momentum: 0.95
  muon_weight_decay: 0.01
  muon_ns_method: gram       # gram 或 standard
  adam_lr: 1.2e-4
  adam_betas: [0.9, 0.999]
  adam_eps: 1.0e-8
  adam_weight_decay: 0.01
  lr_scheduler: constant_with_warmup
  warmup_steps: 1000
```

训练器会在 DeepSpeed 接管模型前为所有参数设置 `use_muon`：二维及以上且名称不含
`embed`/`lm_head` 的可训练参数进入 Muon 组，其余可训练参数进入辅助 AdamW 组。
对于当前 PEFT LoRA 训练，这通常意味着 LoRA A/B 矩阵使用 Muon。两个参数组可以
使用不同的初始学习率；Diffusers scheduler 会按相同比例调度它们。

该模式只允许与脚本创建的 ZeRO-2 plugin 一起使用，不启用 optimizer/parameter
offload，并采用 `reduce_scatter: false` 的保守路径。`accelerator.prepare()` 完成后，
训练器会检查实际 optimizer wrapper 链中是否存在 `MuonWithAuxAdam`；若 DeepSpeed
没有真正接管 Muon 会立即报错，主进程同时打印 wrapper 链和两组可训练参数量。

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
`train/lr`；使用 DeepSpeed Muon 时还会分别记录 `train/lr_muon` 与
`train/lr_adam`（存在辅助 Adam 组时）。micro-step 不会产生重复记录。将
`logging.backend` 改为 `null` 可以关闭在线记录。

## 训练中采样

采样走完整的编辑推理路径：参考图同时经过 Qwen3-VL 与 VAE，参考 latent 在
DiT 内保持 `t=0`，target 从固定 seed 的噪声开始，按 Krea resolution-aware
flow schedule 积分到 `t=0`。采样时关闭 grounding 尺寸抖动，并使用 VAE posterior
mode 编码参考图，因此相同样本和 seed 可以跨 checkpoint 直接比较。

采样使用独立 evaluate 数据集，不进入训练 DataLoader。evaluate manifest 与训练
manifest 使用完全相同的结构，建议从训练数据中排除这些样本。配置中的 `id`
对应 evaluate manifest 顶层的样本 ID：

```yaml
sample:
  enabled: true
  manifest: data/evaluate/manifest.jsonl
  every: 250             # 每 250 个 optimizer update 采样一次
  steps: 20
  guidance_scale: 4.5
  negative_prompt: ""
  schedule_mu: null      # null 使用随输出 token 数变化的 Krea dynamic shift
  samples:
    - id: val-0001
      seed: 42
    - id: val-0002
      seed: 43
```

`sample.every` 按 `global_step` 计算，也就是实际 optimizer update 数；gradient
accumulation 的 micro-step 不计数。每条样本的生成尺寸固定为该条预处理后 GT
target 的尺寸，不提供单独的 `width`/`height` 覆盖；因此最大/最小像素量和 16px
网格对齐只执行一次，生成图与 ground truth 始终具有相同 H/W。单参考与多参考
样本均可采样，多张输入图会渲染成 montage。

每个采样 cell 上方写编辑 prompt，下方为
`[输入参考图 montage | 生成输出图 | ground truth target]`。训练器只根据样本数量
`N` 自动确定接近正方形的 grid：`W = ceil(sqrt(N))`、`H = ceil(N / W)`；不足的
cell 留空。每个采样 step 只保存一张合并图：

```text
output/krea2edit/samples/step-00000250/preview-grid.webp
```

grid 以有损 WebP `quality=80` 保存，避免长期训练的本地与 W&B 图片占用过大。
当 `logging.backend: wandb` 时，每个采样 step 只把这一张 WebP 记录到
`samples/previews`，不会上传各个 cell，也不会额外生成 PNG。

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
- reference-conditioned flow 采样与 WebP 预览：[krea2edit/sampling.py](krea2edit/sampling.py)
- 实验性双向 SVDQuant、转置 checkpoint 与 autograd：[krea2edit/svdquant.py](krea2edit/svdquant.py)
- timestep 策略：[krea2edit/timesteps.py](krea2edit/timesteps.py)
- DiT 与 FA2/FA4 实现：[krea2edit/mmdit.py](krea2edit/mmdit.py)
