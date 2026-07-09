# PSBAF：锅炉灰污预测系统

[English](README.md) | [简体中文](README_zh-CN.md)

PSBAF 是一个面向燃煤电站锅炉灰污状态预测的毕业设计项目。系统集成了小波去噪、变分模态分解（VMD）、TSMixer 与 TiDE 时间序列预测、鲸鱼优化算法（WOA）、消融实验以及基于 Flask 的可视化管理平台。

## 项目简介

锅炉受热面积灰会降低传热效率，并影响机组安全、经济运行。本项目以锅炉清洁因子历史序列为研究对象，通过信号处理和深度学习模型预测其变化趋势，同时比较 VMD、WOA 等模块对预测性能的影响。

系统主要流程如下：

```text
原始清洁因子数据
        │
        ▼
小波阈值去噪
        │
        ▼
VMD 多尺度分解
        │
        ▼
TSMixer / TiDE 时序预测
        │
        ▼
WOA 超参数优化
        │
        ▼
预测结果与消融实验对比
        │
        ▼
Flask 可视化展示
```

## 主要功能

- 数据清洗：采用小波软阈值和硬阈值方法处理清洁因子序列，并计算残差能量比、平滑度等指标。
- 信号分解：使用 VMD 将原始序列分解为多个不同频率尺度的模态分量。
- 时间序列预测：基于 TSMixer 和 TiDE 建立清洁因子预测模型。
- 参数智能优化：使用 WOA 搜索模型超参数，并通过三维散点图展示优化过程。
- 消融实验：比较基线模型、VMD 增强模型、WOA 优化模型以及 VMD+WOA 组合模型。
- 可视化平台：使用 Flask、AdminLTE 和 Pyecharts 展示数据清洗、预测结果、误差指标与优化过程。
- 权限入口：提供普通用户和管理员两类演示账号。

## 系统效果

### TSMixer 消融实验

比较 TSMixer、TSMixer+WOA、VMD+TSMixer 和 VMD+TSMixer+WOA 四种配置的预测曲线及误差。

![TSMixer 消融实验](figures/tsmixer_ablation.png)

### TiDE 消融实验

比较 TiDE、TiDE+WOA、VMD+TiDE 和 VMD+TiDE+WOA 四种配置的预测效果。

![TiDE 消融实验](figures/tide_ablation.png)

### VMD+TSMixer+WOA 预测结果

![VMD+TSMixer+WOA 预测结果](figures/vmd_tsmixer_woa_training.png)

### VMD+TiDE+WOA 预测结果

![VMD+TiDE+WOA 预测结果](figures/vmd_tide_woa_training.png)

## 项目结构

```text
.
├── app.py                         # Flask 系统入口
├── Flask_app.py                   # 旧版 Flask 入口
├── data_cleaning.py               # 小波去噪
├── data_decomposition.py          # VMD 分解可视化
├── data_decomposition_vmd.py      # VMD 封装函数
├── predict_*.py                   # 各类模型训练与预测脚本
├── ablation_study.py              # 中文消融实验绘图
├── ablation_study_english.py      # 英文消融实验绘图
├── charts/                        # Pyecharts 图表生成模块
├── templates/                     # Flask 页面模板
├── static/                        # 前端静态资源
├── figures/                       # 实验结果图片
├── *.csv                          # 预测和优化结果
├── requirements.txt               # Python 依赖
├── README.md                      # 英文说明
└── README_zh-CN.md                # 中文说明
```

## 快速开始

### 1. 创建并激活虚拟环境

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

Linux 或 macOS：

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动系统

```bash
python app.py
```

根据终端提示访问本地地址，Flask 默认地址通常为 `http://127.0.0.1:5000`。

## 演示账号

| 角色 | 用户名 | 密码 |
| --- | --- | --- |
| 普通用户 | `User` | `User` |
| 管理员 | `Admin` | `Admin` |

以上账号仅用于本地项目演示，请勿直接用于生产环境。

## 模型与实验

### 数据预处理

```bash
python data_cleaning.py
```

用于查看清洁因子序列的小波去噪结果。

### VMD 分解

```bash
python data_decomposition.py
```

用于展示 VMD 模态分量及其频率特征。

### 模型预测

```bash
python predict_tsmixer.py
python predict_tide.py
python predict_tsmixer_woa.py
python predict_tide_woa.py
python predict_vmd_tsmixer.py
python predict_vmd_tsmixer_woa.py
python predict_vmd_tide.py
python predict_vmd_tide_woa.py
```

运行结果将保存为 CSV 文件，供可视化平台和消融实验脚本读取。

### 生成实验图片

```bash
python ablation_study_english.py --plot all --save-dir figures --no-show
```

也可以将 `--plot all` 替换为 `tsmixer`、`tide`、`vmd-tsmixer-woa` 或 `vmd-tide-woa`，单独生成对应图片。

## 技术栈

- Python、NumPy、Pandas、SciPy
- PyWavelets、vmdpy
- PyTorch、Darts、Scikit-learn
- TSMixer、TiDE、WOA
- Flask、Flask-Login、Flask-WTF
- AdminLTE、Bootstrap、Pyecharts、Matplotlib

## 数据说明

项目默认不包含原始工业数据文件 `CF1.mat`。如数据允许公开，可将其放入项目根目录；否则建议仅提供脱敏样例数据，并根据实际路径修改 `data_cleaning.py`。

## 声明

本仓库用于毕业设计与作品展示，重点呈现完整的数据处理、建模、优化、实验对比和可视化流程。
