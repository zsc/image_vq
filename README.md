# Open-MAGVIT2 推理工具 (CLI)

这个项目是 Open-MAGVIT2 分词器（Tokenizer）的高级推理工具。它支持图像与离散 Token 之间的双向转换，并具备自适应分辨率选择功能，以在压缩率和图像细节（如文字清晰度）之间取得平衡。

## 功能特点

- **双向转换**：支持 `图像 -> JSON Token` (Encode) 和 `JSON Token -> 图像` (Decode)。
- **自适应分辨率 (Adaptive Res)**：自动进行试错搜索，寻找满足目标 PSNR 的最小分辨率。支持自动 **Upscale** 以满足高质量需求（长边最高支持 2048px）。
- **多格式支持**：支持常见图像格式，并已通过 `pillow-avif-plugin` 扩展支持 **AVIF** 格式。
- **细节保留**：保持原始图像长宽比，通过计算重建图像与原始大图之间的 PSNR，确保细节被充分保留。
- **标准 CLI 界面**：安装后可直接作为系统命令使用。

## 安装步骤

```bash
git clone <repository-url>
cd image_vq
pip install -e .
```

## 使用方法

安装后，你可以直接使用 `image_vq` 命令：

### 1. 图像转 Token (Tokenization)
```bash
# 自动搜索合适尺寸（支持 Upscale）并生成 .json
image_vq 你的图片.png

# 指定目标质量（如 40dB）
image_vq 你的图片.png --psnr 40.0
```

### 2. Token 转图像 (Reconstruction)
```bash
# 自动识别 .json 输入并还原图片
image_vq 你的数据.json -o 还原图.png

# 使用 --avif 标志将输出保存为 .avif 格式
image_vq 你的数据.json --avif
```

## JSON 文件格式设计

Token 文件采用 JSON 格式存储，包含重建图像所需的全部元数据。为了进一步压缩体积，Token 数据被编码为 Base64 字符串：

```json
{
  "original_size": [1672, 2508],  // 原始图像的 [宽, 高]，用于还原
  "resolution": [512, 512],       // Tokenize 时使用的缩放尺寸（可能大于原图）
  "latent_shape": [1, 64, 64],    // 潜空间张量形状 [B, H, W]
  "tokens_b64": "..."             // Base64 编码的二进制 Token 数据
}
```

- **Token 压缩原理**：MAGVIT2-LFQ 将每个空间位置的 18 维二进制向量（+1/-1）压缩为位图（Bitset），然后通过 Base64 编码存储。相比直接存储整数列表，体积减少约 90% 以上。
- **空间效率**：编码时程序会自动计算并显示相对于标准 AVIF 图像的压缩比（例如 `JSON 50KB vs AVIF 150KB (Ratio: 3.0x)`）。

## 参数说明

- `input_path`: 输入文件路径（图片或 .json）。
- `-o, --output`: 输出路径（默认自动生成）。
- `--psnr`: 自动搜索时的目标 PSNR 值（默认 32.0）。
- `--size`: 强制指定固定尺寸（长边），跳过自动搜索。
- `--debug`: 开启后会在 `demo/` 目录生成 `index.html` 报告。
- `--avif`: 指定输出格式为 AVIF。

## 参考
- 模型权重：[TencentARC/Open-MAGVIT2](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-128-resolution)
