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

## 确定性 VAE 与训练分支

训练默认使用 VAE posterior mode，避免 source、target 和 reference 各自采样 posterior
引入额外随机差异。`independent_sample` 仅保留为显式对照选项：

```yaml
train:
  vae_latent_strategy: mode  # mode | independent_sample
```

每个数据样本只选择一个训练分支。配置项是分支的**采样权重**，会在当前样本可用的
分支之间归一化；它们不是 loss multiplier：

```yaml
train:
  branch_sampling_weights:
    edit: 0.75
    instruction_dropout: 0.10
    noop: 0.15
  noop_prompts:
    - "Keep the image exactly unchanged."
    - "Make no changes to the image."
    - "Preserve the image exactly as it is."
```

- `edit`：使用 manifest 中的编辑指令，clean endpoint 是 target latent。
- `instruction_dropout`：使用空指令，clean endpoint 仍是 target latent。
- `noop`：从 `noop_prompts` 均匀随机选择一条指令，clean endpoint 是第一张
  `role: source` reference 的 latent。该 tensor 直接复用 reference 编码结果，不会
  再次执行 VAE 编码。

没有 `role: source` 的样本不能进入 `noop`；训练器会移除该分支，再对剩余分支权重
重新归一化。三种分支共享下述同一组 loss 配置，不增加 branch-specific loss 权重，
也不增加已排除的 edit/no-op delta consistency。

## 可组合 FM、PFM 与 Pixel Loss

PFM 路径参考 [Perceptual Flow Matching 论文](https://arxiv.org/abs/2607.03524)
及其[官方实现](https://github.com/ZhaoChuyang/PFM)。

`train.loss.weights` 中三个系数完全独立；设为 `0` 即禁用，因此可以自由组合纯 FM、
纯 PFM、纯 pixel，或自行决定主项和辅助项：

```yaml
train:
  loss:
    weights:
      fm: 1.0
      perceptual: 0.1
      pixel: 0.1
    pixel_type: mse  # mse | l1 | huber | charbonnier
    huber_delta: 1.0
    charbonnier_epsilon: 0.001
    gradient_checkpointing: true
    perceptual_dtype: bf16  # bf16 | fp32
    perceptual_backbones:
      - name: tipsv2
        model_path: google/tipsv2-b14
        input_size: 448
        min_input_size: 224
        max_input_size: 448
        weight: 0.0
      - name: dinov2
        model_path: facebook/dinov2-base
        input_size: 518
        min_input_size: 224
        max_input_size: 518
        weight: 1.0
      - name: dinov3
        model_path: facebook/dinov3-vits16-pretrain-lvd1689m
        input_size: 512
        min_input_size: 256
        max_input_size: 768
        weight: 0.0
      - name: vgg
        input_size: 224
        min_input_size: 64
        max_input_size: 224
        weight: 1.0
```

所有启用项先使用同一个模型预测。训练器把 ragged velocity tokens 还原为 latent
velocity，并按 Krea convention 计算：

```text
x0_prediction = noisy_target - t * predicted_velocity
```

- `fm`：现有 latent velocity MSE；仅这一项继续使用 timestep loss weighting。
- `perceptual`：可微分地 VAE decode `x0_prediction`，再组合冻结的 TIPSv2、
  DINOv2、DINOv3 和 calibrated VGG LPIPS 特征距离。各 backbone 先乘自己的
  `weight`，其绝对加权和再乘 `weights.perceptual`；内部权重不会自动归一化。
- `pixel`：在 `[-1, 1]` RGB 空间比较同一张 decode 预测与监督图，支持 MSE、L1、
  Huber 和 Charbonnier。

按当前实现，监督端直接使用 VAE encode 之前的数据图像，不额外 decode clean
latent：edit 与 instruction-dropout 使用原 target，noop 使用 primary source。预测端
VAE decode 必须保留梯度，但 VAE 参数仍冻结；`gradient_checkpointing: true` 会对预测
decode 和 perceptual encoder 做 activation checkpoint 以降低峰值显存。PFM/pixel
不会再乘 FM 的 timestep weight。预测 decode 与 encode 前监督图的 C/H/W 必须完全
一致，否则训练立即报错；loss 路径不会通过 resize 或 padding 隐式兜底。

DiT 数据预处理先按自己的最小/最大像素预算确定训练画布；感知 loss 不复用该预算。
每个感知 backbone 在 forward 前对 DiT 输出做第二次尺寸检查：若 H/W 已位于自己的
`min_input_size` 与 `max_input_size` 内就保留原尺度，否则使用一个共同 scale 等比
放大或缩小，令两条边都进入该闭区间。整数范围同时约束 H/W，也可写成
`[height, width]` 分别约束两轴。若极端长宽比不存在能同时满足两轴范围的等比 scale，
会立即报错，不会拉伸图像或裁掉边缘。

等比缩放后，H/W 再分别取到最近的模型 patch 网格（TIPSv2/DINOv2 通常为 14，
DINOv3 ViT 通常为 16）；取整结果仍保证处于范围内。prediction 与 target 始终共用
同一个最终尺寸。VGG 没有 ViT patch 边界。

每个 backbone 都可独立设置 `input_size`：整数表示固定 `N×N`，
`[height, width]` 表示固定矩形尺寸。设置后不再执行等比范围拟合，而是把 prediction
与 target 一起直接双线性 resize 到该尺寸，因此方形尺寸会拉伸非方形输入；删除某个
条目的 `input_size` 后，只有该模型恢复上述动态范围模式。固定尺寸仍须位于该模型的
`min_input_size` / `max_input_size` 内，并且必须已经对齐其 patch 网格；配置不满足时
直接报错，不会静默修改配置值。范围检查和 prediction/target 同形检查之外，模型不
支持的配置会由其原生 forward 直接报错，不做额外兜底。

每个样本的 prediction/target 原始 tensor shape 会在任何 resize 之前逐对检查。检查
通过后才按各 backbone 的最终输入尺寸分组并拼成 batch；固定 `input_size` 时整个
micro-batch 对该 backbone 只执行一次编码。动态范围模式若解析出多个不同 H/W，则按
H/W 分组，每个尺寸组执行一次编码，不会在训练循环中逐图调用视觉 backbone。

`name` 支持 `tipsv2`、`dinov2`（别名 `dino2`）、`dinov3`（别名 `dino3`）和
`vgg`，可按任意顺序组合；`weight: 0` 的条目完全不会加载。DINOv2 保留公开 PFM
实现的全层 normalized feature MSE。DINOv3 使用同一全层距离，但只比较 spatial
patch tokens；TIPSv2 使用最终 spatial patch tokens。后两项是本项目将
视觉 backbone 用作感知特征的适配，不应理解为上游发布的校准 perceptual metric。

内置范围以具体权重见过的训练 crop 为依据，而不是模型结构能够前向的极限：

- TIPSv2 全系列均为 patch 14。低分辨率阶段使用 global/local crop 224/98，高分辨率
  阶段使用 448/140，蒸馏学生也经历高分辨率阶段。感知 loss 使用整图 spatial
  features，所以默认采用 global crop 范围 224--448。
- `facebook/dinov2-base` 的主训练 global/local crop 是 224/98，发布权重随后在 518
  做高分辨率适配（patch 14）。公开 PFM recipe 也在 518 提取 DINOv2 特征，因此默认
  global 范围是 224--518。
- `facebook/dinov3-vits16-pretrain-lvd1689m` 的主训练/蒸馏 global/local crop 是
  256/112（patch 16）。论文描述的蒸馏学生随后也执行高分辨率阶段（不使用 Gram
  anchoring），global crop 为 512 或 768，所以默认范围是 256--768。
- VGG LPIPS 的线性校准使用 64px BAPPS patch，底层 VGG16 的 ImageNet 训练输入为
  224px，因此该组合默认范围是 64--224。

这些默认上下限会实际参与运行时检查。自定义 checkpoint 如果使用了不同训练范围，
可在对应条目显式覆盖 `min_input_size` / `max_input_size`；`input_size` 本身也必须落在
覆盖后的范围内。
DINOv3 patch-16 在 256/512/768 下分别产生 256/1024/2304 个 spatial tokens，完整
self-attention 的矩阵规模约为 1/16/81 倍，因此感知 loss 使用 512 或 768 时显存和
计算量会明显上升。

感知网络精度独立于 DiT，默认使用 BF16；如需更高数值精度可设为 FP32，不允许使用
容易令小特征差下溢的 FP16。

示例中的 `1.0 / 0.1 / 0.1` 是混合目标的保守起点，并不是 PFM 论文给出的混合权重。
论文原方法用 PFM 替换 FM，而不是把它作为辅助项。把 `perceptual` 设为非零时，首次
运行需要预先缓存所有非零权重条目。TIPSv2 默认模型为 `google/tipsv2-b14`；其当前
Hugging Face 实现通过 `trust_remote_code=True` 加载，但 loss 直接调用冻结的 vision
tower 以保留对 prediction 的梯度，并释放不使用的 text tower。DINOv3 默认模型
`facebook/dinov3-vits16-pretrain-lvd1689m` 是 gated 权重，需要先接受其许可并缓存。
当前启动脚本使用 Hugging Face 离线模式，因此缺失缓存时会直接报错。只需要 pixel
loss 时可把 `perceptual` 设为 `0`，此时不会加载任何感知模型。

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
