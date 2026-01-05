# Open-MAGVIT2 推理工具 (CLI)

这个项目是 Open-MAGVIT2 分词器（Tokenizer）的高级推理工具。它支持图像与离散 Token 之间的双向转换，并具备自适应分辨率选择功能，以在压缩率和图像细节（如文字清晰度）之间取得平衡。

## 功能特点

- **双向转换**：支持 `图像 -> JSON Token` (Encode) 和 `JSON Token -> 图像` (Decode)。
- **自适应分辨率 (Adaptive Res)**：自动进行试错搜索（Trial-and-error），寻找满足目标 PSNR（默认 32dB）的最小分辨率。
- **细节保留**：通过计算重建图像与原始大图之间的 PSNR，确保文字等细节在特定尺寸下被充分保留。
- **标准 CLI 界面**：安装后可直接作为系统命令使用。
- **调试模式**：仅在开启 `--debug` 时生成可视化 HTML 报告。

## 安装步骤

```bash
git clone <repository-url>
cd Open-Magvit2-demo
pip install -e .
```

## 使用方法

安装后，你可以直接使用 `open-magvit2-demo` 命令：

### 1. 图像转 Token (Tokenization)
```bash
# 自动搜索合适尺寸并生成 .json
open-magvit2-demo 你的图片.png

# 开启调试模式并指定目标质量
open-magvit2-demo 你的图片.png --target-psnr 35.0 --debug
```

### 2. Token 转图像 (Reconstruction)
```bash
# 自动识别 .json 输入并还原图片
open-magvit2-demo 你的数据.json -o 还原图.png
```

## JSON 文件格式设计

Token 文件采用 JSON 格式存储，包含重建图像所需的全部元数据：

```json
{
  "original_size": [1672, 2508],  // 原始图像的 [宽, 高]，用于还原
  "resolution": [512, 512],       // Tokenize 时使用的缩放尺寸
  "latent_shape": [1, 64, 64],    // 潜空间张量形状 [B, H, W]
  "tokens": [123, 456, ...]       // 扁平化的整数 Token 序列 (0 ~ 262143)
}
```

- **Token 压缩原理**：MAGVIT2-LFQ 将每个空间位置的 18 维二进制向量（+1/-1）压缩为一个整数索引（$2^{18} = 262144$）。
- **空间效率**：相比存储原始像素，JSON 格式记录的 Token 极大地降低了数据量。

## 参数说明

- `input_path`: 输入文件路径（图片或 .json）。
- `-o, --output`: 输出路径（默认自动生成）。
- `--target-psnr`: 自动搜索时的目标 PSNR 值（默认 32.0）。
- `--size`: 强制指定固定尺寸，跳过自动搜索。
- `--debug`: 开启后会在 `demo/` 目录生成 `index.html` 报告。

## 参考
- 模型权重：[TencentARC/Open-MAGVIT2](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-128-resolution)
