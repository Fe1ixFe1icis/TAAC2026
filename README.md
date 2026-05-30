<div align="center">

# 🎯 TAAC 2026 CTR 预估模型

[![AUC](https://img.shields.io/badge/AUC-0.7741-brightgreen?style=for-the-badge&logo=target)](https://github.com/Fe1ixFe1icis/TAAC2026)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**统一 Token 流架构的点击率预估方案**


</div>

---

## 📋 目录

- [项目概述](#-项目概述)
- [模型架构](#-模型架构)
- [核心实验结果](#-核心实验结果)
- [优化历程](#-优化历程)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [参考文献](#-参考文献)
- [致谢](#-致谢)

---

## 🎯 项目概述

本项目是 **TAAC 2026**（腾讯广告算法大赛）的参赛方案，采用创新的 **TokenFormer** 架构进行点击率（CTR）预估。

### 🏆 最佳成绩

| 指标 | 数值 | 配置 |
|------|------|------|
| **AUC** | **0.7741** | 10K 步, d_model=64, gate_bias=2.0 |
| LogLoss | 0.4725 | AMP BF16, batch_size=512 |
| 训练时间 | ~28 分钟 | NVIDIA GPU, CUDA 12.6 |
| 参数量 | 130.4M | 稀疏: 129.3M, 密集: 1.04M |

### ✨ 核心创新点

- **统一 Token 流**: 所有特征（静态、连续、序列）统一表示为 Token 序列
- **逐域 Token 化**: 每个特征域独立 Token，带残差 SwiGLU + 门控
- **BFTS 注意力掩码**: 双向-域-时序-分隔符掩码，实现结构化交互
- **滑动窗口注意力**: 深层高效注意力机制
- **OneTrans 混合参数**: F-Token 异构参数，T-Token 共享参数

---

## 📊 模型架构

### TokenFormer 模块

```
┌─────────────────────────────────────────┐
│         TokenFormer 模块                 │
├─────────────────────────────────────────┤
│  输入: X ∈ ℝ^(B×N×D)                     │
│                                         │
│  ┌─────────────┐                        │
│  │  RMSNorm    │                        │
│  └──────┬──────┘                        │
│         ▼                               │
│  ┌─────────────┐   ┌─────────────┐     │
│  │  W_q, W_k   │   │  NLIR 门控  │     │
│  │  W_v, W_o   │   │  W_g        │     │
│  └──────┬──────┘   └──────┬──────┘     │
│         │                  │            │
│         ▼                  ▼            │
│  ┌─────────────────────────────────┐   │
│  │  缩放点积注意力                  │   │
│  │  + RoPE + BFTS 掩码             │   │
│  └─────────────────────────────────┘   │
│         │                               │
│         ▼                               │
│  ┌─────────────┐   ┌─────────────┐     │
│  │  gate * A   │ + │  残差连接   │     │
│  └─────────────┘   └─────────────┘     │
│         │                               │
│         ▼                               │
│  ┌─────────────┐                        │
│  │  RMSNorm    │                        │
│  └──────┬──────┘                        │
│         ▼                               │
│  ┌─────────────┐                        │
│  │  SwiGLU FFN │                        │
│  └──────┬──────┘                        │
│         │                               │
│         ▼                               │
│  输出: X' = I + FFN(I)                  │
└─────────────────────────────────────────┘
```

### 统一 Token 流

```
[F1, F2, ..., F_n, SEP, T1, T2, ..., T_m, SEP, V]
 │   │         │    │   │   │         │    │  │
 │   │         │    │   │   │         │    │  └── 目标 Token (CTR 预估)
 │   │         │    │   │   │         │    └───── 分隔符
 │   │         │    │   │   │         └────────── 序列 Token
 │   │         │    │   │   └──────────────────── (用户行为历史)
 │   │         │    │   └────────────────────────
 │   │         │    └──────────────────────────── 分隔符
 │   │         └───────────────────────────────── 静态特征 Token
 │   └─────────────────────────────────────────── (user_int, item_int)
 └───────────────────────────────────────────────
```

---

## 🚀 核心实验结果

### 实验汇总

| 实验 | AUC | LogLoss | 时间 | 关键配置 | 对比基线 |
|:-----------|:---:|:-------:|:----:|:-----------|:-----------:|
| **E6 (最佳)** | **0.7741** | **0.4725** | 28:24 | 10K 步, gate_bias=2.0 | **+0.78%** |
| E4 (OneTrans) | 0.7689 | 0.4761 | 29:40 | mixed_params=True | +0.10% |
| E2 (SWA 大窗口) | 0.7682 | 0.4767 | 13:14 | SWA=[128,64] | +0.01% |
| E1 (基线) | 0.7681 | 0.4767 | 13:09 | 5K 步, d_model=64 | — |
| E3 (SWA 小窗口) | 0.7681 | 0.4767 | 13:10 | SWA=[32,16] | 0.00% |
| E5 (门控 v1) | 0.7680 | 0.4768 | 14:27 | gate_bias=0.0 | -0.01% |

### 优化洞察

```
AUC 提升时间线
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基线 (E1)         ████████████████████████████████████  0.7681
+ AMP BF16        ████████████████████████████████████  0.7681  (速度 ↑)
+ SWA 调参        ████████████████████████████████████░ 0.7682  (+0.01%)
+ OneTrans        ████████████████████████████████████░ 0.7689  (+0.10%)
+ 10K 步          █████████████████████████████████████ 0.7741  (+0.78%) ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 关键发现

1. 🎯 **训练时长最关键**: 10K 步 vs 5K 步 → **+0.0060 AUC**
2. 🔓 **门控初始化至关重要**: bias=2.0 (sigmoid≈0.88) 优于 bias=0.0
3. 📐 **SWA 窗口影响微弱**: [32,16] vs [64,32] vs [128,64] → <0.0001 差异
4. ⚖️ **OneTrans 混合参数**: +0.0007 AUC 但训练时间翻倍
5. 🚫 **d_model=128 失败**: 显存不足，35秒/步 (对比 d=64 的 6秒/步)

---

## 📈 优化历程

### 阶段 A: 训练效率（快速收益）
- ✅ AMP BF16 混合精度
- ✅ Batch size 512（从 256 提升）
- ✅ SWA 窗口调参

### 阶段 B: 结构改进
- ✅ OneTrans 混合参数机制
- ✅ 逐域门控 + 改进初始化
- ⚠️ d_model 扩展放弃（显存限制）

### 阶段 C: 训练时长
- ✅ 扩展到 10K 步
- ✅ 门控偏置初始化 = 2.0
- 🏆 **最佳 AUC: 0.7741**

### 阶段 D: 未来方向（来自 TokenMixer-Large 论文）
- 🔄 全局 Token（CLS 风格聚合）
- 🔄 跨层残差（深层模型）
- 🔄 稀疏逐域 MoE
- 🔄 FP8 量化

---

## 🔧 Quick Start

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/Fe1ixFe1icis/TAAC2026.git
cd TAAC2026

# 设置环境（推荐 uv）
uv sync --locked --extra dev --extra cuda126

# 或使用 pip
pip install -e ".[dev,cuda126]"
```

### Training

```bash
# 最佳配置 (AUC=0.7741)
uv run taac-train \
  --experiment win_version/experiments/tokenformer \
  --dataset-path data/merged \
  --schema-path data/merged/schema.json \
  --run-dir outputs/tokenformer_best \
  --max-steps 10000 \
  --batch-size 512 \
  --device cuda \
  --amp \
  --amp-dtype bfloat16
```

### Evaluation

```bash
uv run taac-eval \
  --checkpoint outputs/tokenformer_best/global_step10000.AUC=0.774144 \
  --dataset-path data/merged \
  --split validation
```

---

## 📁 Structure

```
TAAC2026/
├── 📁 src/taac2026/              # 核心框架
│   ├── domain/                   # 配置、Schema、接口
│   ├── application/              # 训练、评估、推理流程
│   └── infrastructure/           # 数据管道、建模、加速
│
├── 📁 win_version/experiments/   # 实验包
│   ├── tokenformer/              # Best Model
│   │   ├── __init__.py           # 实验配置
│   │   └── model.py              # TokenFormer 实现
│   └── baseline/                 # PCVRHyFormer 基线
│
├── 📁 docs/                      # 文档
│   ├── architecture.md
│   └── guide/
│
├── 📁 tests/                     # 单元、契约、集成测试
│
├── 📄 win_version/logs.html      # 详细实验日志
├── 📄 win_version/optimization_proposal.html  # 优化分析
├── 📄 README.md                  # 本文件
├── 📄 pyproject.toml             # 依赖管理
└── 📄 run.sh                     # 入口脚本
```

---

## 📚 参考文献

### 工作基于

1. **TokenMixer-Large** (ByteDance, 2026)
   - *Scaling Up Large Ranking Models in Industrial Recommenders*
   - arXiv:2602.06563v2 [cs.IR]
   - 核心洞察: Mixing & Reverting, Sparse Per-token MoE

2. **RankMixer** (ByteDance)
   - 面向排序的 TokenMixer 骨干网络
   - 硬件协同设计实现高 MFU

3. **PCVRHyFormer** (TAAC 基线)
   - 多数据集 CTR 预估框架
   - NS Tokenizer, RoPE, Flash Attention

### 相关研究

- **HSTU** (Meta, 2024) — 层次化序列转导单元
- **DHEN** (Google) — 深度层次集成网络
- **Wukong** (ByteDance) — 推荐系统 Scaling Law
- **LONGER** (ByteDance) — 超长期兴趣建模

---

<div align="center">

[⭐ 收藏本仓库](https://github.com/Fe1ixFe1icis/TAAC2026) • [🐛 报告问题](../../issues) • [💡 功能建议](../../issues)

</div>
