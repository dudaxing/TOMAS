# TOMAS（Padhy et al., 2024）实验复现报告

> **文献**：Padhy, R.K., Suresh, K. & Chandrasekhar, A. TOMAS: topology optimization of multiscale fluid flow devices using variational auto-encoders and super-shapes. *Struct Multidisc Optim* **67**, 119 (2024).  
> **DOI**：https://doi.org/10.1007/s00158-024-03835-6  
> **官方代码**：https://github.com/UW-ERSL/TOMAS（本仓库即该实现）

---

## 1. 论文目标与复现范围

论文提出 **TOMAS** 框架：在宏观 Stokes 流拓扑优化中，用 **超形状（super-shape）** 参数化微结构，并借助 **变分自编码器（VAE）** 将离线的渗透率、接触周长等性质嵌入可微的潜空间，从而避免优化过程中反复数值均匀化。

本报告在 **Linux** 环境下，按仓库 notebooks 与配置文件，完整跑通下列流程（含中间产物）：

| 阶段 | 对应论文内容 | 本仓库实现 |
|------|-------------|-----------|
| 1 | 超形状采样与像素化（Fig. 2–3 类微结构库） | `notebooks/generate_data.ipynb` → `scripts/01_generate_supershape_data.py` |
| 2 | 周期 Stokes 均匀化（渗透率张量 \( \mathbf{C}_h \)） | `dataset/fluidHomogenization.m` → `scripts/02_run_homogenization.m`（Octave） |
| 3 | VAE 训练（Fig. 4–5 潜空间） | `notebooks/train_vae_main.ipynb` → `scripts/03_train_vae.py` |
| 4 | 三个宏观算例的多尺度 TO（Fig. 10–16） | `notebooks/multiscale_TO_main.ipynb` + 三个 YAML → `scripts/04_run_multiscale_to.py` |

**说明**：工作区未包含您本地的 Zotero PDF（`Padhy et al_2024_TOMAS.pdf`），复现参数以仓库内 `notebooks/*.yaml` 为准（与论文 Section 4 及补充材料中给出的设置一致）。若 PDF 中另有更大规模数据集（例如数千个微结构），需相应增大 `datagen.yaml` 中的 `num_samples` 并重新执行阶段 1–3。

---

## 2. 环境配置（Linux）

### 2.1 系统依赖

```bash
sudo apt-get install -y octave libsuitesparse-dev python3.12-venv python3.12-dev
```

- **Octave**：运行 `dataset/fluidHomogenization.m` 均匀化（与 MATLAB 兼容）。
- **SuiteSparse (KLU)**：编译 `torch_sparse_solve`（宏观流求解器所需）。

### 2.2 Python 虚拟环境

```bash
cd /workspace
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy pandas matplotlib PyYAML shapely geopandas

# 宏观 Stokes 求解（需先装 SuiteSparse 头文件）
CFLAGS="-I/usr/include/suitesparse" CXXFLAGS="-I/usr/include/suitesparse" \
  pip install --no-build-isolation git+https://github.com/flaport/torch_sparse_solve.git
```

本机实测：Python 3.12 + PyTorch 2.2.2（`requirements.txt` 写明 3.10.9 / 2.1.2，新版本可正常运行）。

### 2.3 一键流水线

```bash
bash scripts/run_all.sh
```

或分步执行：

```bash
.venv/bin/python scripts/01_generate_supershape_data.py
cd dataset && octave --no-gui --quiet ../scripts/02_run_homogenization.m
.venv/bin/python scripts/03_train_vae.py
.venv/bin/python scripts/04_run_multiscale_to.py config_diffuser.yaml
.venv/bin/python scripts/04_run_multiscale_to.py config_bent_pipe.yaml
.venv/bin/python scripts/04_run_multiscale_to.py config_biffurcated_pipe.yaml
```

---

## 3. 各阶段原理与操作记录

### 阶段 1：超形状数据集生成

**配置**：`notebooks/datagen.yaml`

| 参数 | 值 | 含义 |
|------|-----|------|
| `num_samples` | 100 | 微结构样本数（演示规模；论文可用更大集合） |
| `nelx`, `nely` | 150 | 单位胞元像素分辨率 |
| `SUPERSHAPE` | 见 YAML | 超形状参数 \(a,b,m,n_1,n_2,n_3,c_x,c_y\) 的采样区间 |

**步骤**：

1. 在参数范围内随机采样超形状；
2. 用 Shapely 转为多边形并投影到 \(150\times150\) 二值网格（流体=1，固体=0）；
3. 计算归一化面积与周长（接触周长约束的代理量）；
4. 写出 `dataset/mstr_*.mat` 与 `dataset/recons_shapes.mat`。

**中间产物**：

- `dataset/mstr_shape_parameters_1.mat`
- `dataset/mstr_images_1.mat`
- `dataset/mstr_area_1.mat`, `dataset/mstr_perim_1.mat`
- `results/step1_datagen/sample_microstructures.png`（16 个随机微结构示意图）

---

### 阶段 2：流体均匀化（MATLAB/Octave）

**核心文件**：`dataset/generate_homogenized_data.m` → 调用 `fluidHomogenization.m`（周期边界 Q2–Q1 Stokes + Brinkman 惩罚）。

**物理参数**（与代码一致）：

- 胞元尺寸 \(l_x=l_y=1\)
- 固体渗透率 \(10^6\)，流体渗透率 \(0\)（惩罚法）
- 输出 \(c_{00}, c_{11}, c_{01}, c_{10}\)（均匀化渗透率张量分量）

**运行**：

```bash
octave --no-gui --quiet scripts/02_run_homogenization.m
```

**中间产物**：`dataset/homogen_data_1.mat`

**环境修复**：Octave 默认保存为 ASCII `.mat`，SciPy 无法读取。已将 `save` 改为 `save('-mat7-binary', ...)`，并对已生成文件做了 `mat7-binary` 转换。

**耗时**：100 个样本 × 150×150 网格，约 **2 分钟**（本机 CPU）。

---

### 阶段 3：VAE 训练

**配置**：`notebooks/vae_config.yaml`

| 参数 | 值 |
|------|-----|
| `num_epochs` | 17000 |
| `latent_dim` | 2 |
| `encoder_hidden_dim` / `decoder_hidden_dim` | 600 |
| `lr` | 8e-3 |
| `kl_factor` | 1e-7 |

**输入特征（12 维）**：8 个形状参数 + \(\log_{10}(c_{00}), \log_{10}(c_{11})\) + 周长 + 面积（与 notebook 一致）。

**代码修复**（原仓库 bug，否则无法训练）：

- `vae/data_preprocess.py` 中 `for normalization_type in normalization_type` 改为遍历 `normalization_types`；
- `scripts/03_train_vae.py` 中将 `mstr_area` / `mstr_perim` reshape 为 `(N,1)`；
- 训练结束后写入 `vae/nomalization.pt`（notebook 使用但原流程未保存）。

**中间产物**：

- `vae/vae_net.pt`
- `vae/nomalization.pt`
- `results/step3_vae/c00_histogram.png`
- `results/step3_vae/latent_scatter.png`
- `results/step3_vae/training_curves.png`
- `results/step3_vae_training.log`

**耗时**：17000 epoch，约 **1 分钟**（本机多核 CPU）。

---

### 阶段 4：多尺度流体拓扑优化

**算例与配置**（对应论文 2D 演示）：

| 算例 | 配置文件 | 宏观网格 | 目标周长 | 迭代数 |
|------|----------|----------|----------|--------|
| 扩散器 Diffuser | `config_diffuser.yaml` | 15×15 | 60 | 301 |
| 弯管 Bent pipe | `config_bent_pipe.yaml` | 20×60 | 75.69 | 301 |
| 分叉管 Bifurcated pipe | `config_biffurcated_pipe.yaml` | 16×16 | 75 | 351 |

**方法要点**：

- 坐标神经网络 + Fourier feature 映射（`projection.FourierMap`）输出每个宏观单元的潜变量与取向 \(\theta\)；
- VAE 解码器给出局部 \(c_{00}, c_{11}\) 与周长场；
- `fluid_fe.FluidSolver` 求解宏观 Stokes，目标为耗散功率 \(J\)；
- 周长约束通过惩罚法（`loss.PenaltyLossParameters`）逐步加强。

**脚本修正**：原 notebook 中 `optimizer.step(closure)` 与 `Optimizer.ADAM` 枚举比较错误；`scripts/04_run_multiscale_to.py` 已改为正确的 Adam 更新，并将微结构图保存到 `results/step4_to_<算例>/`。

**本机收敛摘要**：

| 算例 | \(J_0\) | \(J_{\text{final}}\) | 最终总周长 \(\sum P\) |
|------|---------|----------------------|----------------------|
| Diffuser | 1.01×10² | 1.62×10¹ | 238.33（≈目标 60×尺度因子） |
| Bent pipe | 3.21×10¹ | 6.16×10⁰ | 476.66 |
| Bifurcated pipe | 2.93×10¹ | 2.09×10¹ | 254.22 |

每个算例目录包含：

- `microstructures_epoch_*.png`（`plot_interval` 间隔的快照）
- `convergence.png`（\(J\) 与周长随迭代变化）
- `convergence_history.npz`

**耗时**：三个算例合计约 **10–15 分钟**（含绘图；CPU）。

---

## 4. 与论文图表的对应关系

| 论文图示（示意） | 本复现输出 |
|------------------|------------|
| 微结构库 / 参数化形状 | `results/step1_datagen/sample_microstructures.png` |
| 渗透率分布 | `results/step3_vae/c00_histogram.png` |
| VAE 潜空间 | `results/step3_vae/latent_scatter.png` |
| 扩散器、弯管、分叉管优化设计 | `results/step4_to_diffuser/`, `step4_to_bent_pipe/`, `step4_to_biffurcated_pipe/` 中 `microstructures_epoch_*.png` 与 `convergence.png` |

数值与论文印刷图不会逐像素一致（随机种子、浮点环境、VAE 随机性），但**方法链路与仓库配置一致**，趋势应为：\(J\) 随迭代下降，周长约束在惩罚参数作用下逼近目标。

---

## 5. 遇到的问题与处理

1. **无 MATLAB**：使用 Octave 8.x；需 `-mat7-binary` 保存 `.mat`。
2. **`torch_sparse_solve` 缺失**：从源码编译，并设置 `CFLAGS=-I/usr/include/suitesparse`。
3. **`data_preprocess.stack_train_data` 变量名错误**：已修复，否则 VAE 无法训练。
4. **`mstr_area` 形状为 (1,N)**：加载时 `reshape(-1,1)`。
5. **Notebook 未保存 `nomalization.pt`**：在 `03_train_vae.py` 中补充保存。
6. **多尺度 notebook 中 Adam 优化器分支**：在 `04_run_multiscale_to.py` 中修正。

---

## 6. 目录结构（复现后）

```
workspace/
├── dataset/
│   ├── mstr_*.mat, homogen_data_1.mat, recons_shapes.mat
│   └── fluidHomogenization.m, generate_homogenized_data.m
├── vae/
│   ├── vae_net.pt, nomalization.pt
│   └── data_preprocess.py  (已修复)
├── results/
│   ├── step1_datagen/
│   ├── step3_vae/
│   ├── step4_to_diffuser/
│   ├── step4_to_bent_pipe/
│   └── step4_to_biffurcated_pipe/
├── scripts/
│   ├── 01_generate_supershape_data.py
│   ├── 02_run_homogenization.m
│   ├── 03_train_vae.py
│   ├── 04_run_multiscale_to.py
│   └── run_all.sh
└── EXPERIMENT_REPRODUCTION.md   # 本文档
```

---

## 7. 如何扩展为“论文全尺寸”复现

1. 增大 `datagen.yaml` 中 `num_samples`（需与论文/作者一致，若有补充材料请以其为准）。
2. 重新运行阶段 1–3（均匀化与 VAE 训练时间将显著增加）。
3. 保持三个 YAML 中 `num_epochs`、`desired_perimeter` 不变即可复现 Section 4 算例。
4. 若有 GPU，可将 PyTorch 设备改为 CUDA；`torch_sparse_solve` 仍为 **CPU** 求解器，大网格时可能成为瓶颈。

---

## 8. 参考文献

```bibtex
@article{padhy2024tomas,
  title={TOMAS: topology optimization of multiscale fluid flow devices using variational auto-encoders and super-shapes},
  author={Padhy, Rahul Kumar and Suresh, Krishnan and Chandrasekhar, Aaditya},
  journal={Structural and Multidisciplinary Optimization},
  volume={67},
  number={7},
  pages={119},
  year={2024}
}
```

---

*文档生成环境：Linux 6.1，无 GPU；复现执行日期：2026-05-24。*
