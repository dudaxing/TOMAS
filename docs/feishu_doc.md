# TOMAS 论文复现：过程、理论、建模与代码实现

> 论文：Padhy, Suresh, Chandrasekhar. **TOMAS: topology optimization of multiscale fluid flow devices using variational auto-encoders and super-shapes.** *Structural and Multidisciplinary Optimization* (2024) 67:119. DOI: 10.1007/s00158-024-03835-6
>
> 原始代码：https://github.com/UW-ERSL/TOMAS ｜ 复现代码：https://github.com/dudaxing/TOMAS （分支 `pure-claude`）

本文档系统记录在 **WSL (Ubuntu-22.04)** 环境下从零复现该论文全部结果的完整过程，重点说明：(1) 复现中所依据的**理论基础**；(2) 物理与几何**建模**方法；(3) 把原仓库 **MATLAB 流体均匀化改写为 Python 并行实现**的**代码编写细节**；(4) 复现结果与论文的对比及改进。

---

## 1. 总体思路与复现目标

TOMAS 是一个**多尺度流体拓扑优化（multiscale fluid topology optimization）**框架。工程目标：在满足接触面积（2D 中即微结构周长总和）约束的前提下，**最小化流体在器件中的耗散功率**。它把"宏观流道布局"与"微观多孔微结构"统一到一个可微优化问题中，三个阶段如下。

1. **离线均匀化（Algorithm 1）**：用 super-shape（Gielis 超公式曲线）参数化微结构，随机采样 7000 个微结构，在 150×150 网格上做 Stokes/Brinkman 数值均匀化，得到 2×2 渗透率张量 `C`、接触面积 Γ、体积分数 vf。**这部分原仓库是 MATLAB 实现，是本复现的核心移植对象。**
2. **VAE 训练（Algorithm 2）**：在 12 维属性（8 个 super-shape 形状参数 + C00 + C11 + 周长 + 面积）上训练变分自编码器，把离散、高维、不规则的微结构库压缩成一个 **2 维、连续、可微**的隐空间（latent space）。
3. **全局优化（Algorithm 3）**：用坐标神经网络（coordinate network）输出每个宏观单元的隐变量 (z₁, z₂) 与微结构取向 θ，经 VAE 解码器映射到局部渗透率张量，组装 Taylor–Hood (Q2-Q1) 有限元 Stokes 方程求解流场，再用 PyTorch 自动微分对网络参数做基于罚函数的梯度优化。

### 需要复现的结果清单

| 论文位置 | 内容 | 对应脚本 |
|---|---|---|
| Fig 7 | 隐空间密度分布 | `scripts/run_latent_space.py` |
| Table 1 | 重构精度（数据集内/外点） | `scripts/run_latent_space.py` |
| §3.1 / Fig 10 | 理想微结构选取（vf≈0.25，最大 trace(C)）→ M* | `scripts/run_latent_space.py` |
| §3.2 / Fig 11 | 弯管，仅优化取向，耗散功率≈15.1 | `scripts/run_to.py --fix-latent` |
| §3.3 / Fig 12 | 弯管，微结构变化：体积约束(P≈9.61)/周长约束(P≈7.56) | `scripts/run_to.py` |
| §3.4 / Fig 13 | 扩散器收敛过程（接触面积 60） | `scripts/run_to.py` |
| §3.5 / Fig 14-15 | Pareto 前沿 + 速度场 | `scripts/run_pareto.py` |
| §3.6 | 计算成本 | 各脚本计时 |
| §3.7 / Fig 16 | 分叉管（可制造性，接触面积 70） | `scripts/run_to.py` |

---

## 2. 理论基础（复现中考虑的核心理论）

本节梳理复现时反复推敲的物理与数学理论，是正确移植代码、判断结果合理性的依据。

### 2.1 多孔介质中的 Stokes 流与 Brinkman 罚项

器件内的低雷诺数流动由 **Stokes 方程**控制（忽略惯性项）：

```
-μ ∇²u + ∇p = f,      ∇·u = 0
```

其中 u 为速度，p 为压力，μ 为动力黏度，f 为体力。为了在同一套有限元里同时表达"固体（不可流）"与"流体（自由流）"，TOMAS 采用 **Brinkman 罚项**把固体区域建模为渗透率极低的多孔介质：

```
-μ ∇²u + α(x) u + ∇p = f
```

- α(x) 为**逆渗透率（inverse permeability）**：流体区 α=0，固体区 α=αₘₐₓ（取 1e6）。
- 当 α 很大时，动量方程被 α·u 主导，强制该处 u→0，从而"关闭"固体像素的流动。这把"拓扑"（哪里能流）转化为一个**连续材料场**，使梯度优化可行。

### 2.2 周期性数值均匀化：从微结构图像到渗透率张量 C

单个微结构（一个周期性单胞）的宏观等效渗透率，通过**渐近均匀化（asymptotic homogenization）**得到。对单胞施加**周期性边界条件**，分别加两个单位体力载荷（f=(1,0) 与 f=(0,1)）求解上述 Brinkman-Stokes 方程，单胞内速度的体积平均给出 2×2 渗透率张量：

```
C_ij = ⟨u_i⟩  (在第 j 个单位体力载荷下，对单胞体积取平均)
```

- 对角项 C00、C11 表征沿 x、y 的导流能力；非对角项 C01、C10 对各向异性微结构非零，但对对称形状≈0。
- 数值实现基于 Andreassen & Andreasen (2014) 的均匀化框架，针对流体改写为 4 节点双线性、**压力稳定化（pressure-stabilized）**单元。这是本复现移植的关键算法（见 §4）。

### 2.3 super-shape 几何参数化（建模微结构）

微结构边界用 **Gielis 超公式（super-formula）**生成，极坐标下：

```
r(φ) = [ |cos(mφ/4)/a|^n2 + |sin(mφ/4)/b|^n3 ]^(-1/n1)
```

8 个参数 `[a, b, m, n1, n2, n3, cx, cy]` 控制对称性、棱角、胖瘦与中心位置。优点：用少量连续参数即可覆盖从圆、方、星形到花瓣等丰富拓扑，且天然适合做参数采样与 VAE 学习。

### 2.4 变分自编码器（VAE）：构造可微设计空间

直接在 super-shape 参数 + 渗透率的 12 维属性空间上做拓扑优化是不可行的（离散采样、维度高、约束复杂）。VAE 把它学习成一个低维连续流形：

- **编码器** q(z|x) 把 12 维属性映射到 2 维隐分布的 (μ, σ)；
- **重参数化** z = μ + σ·ε, ε∼N(0,I)，使采样可微（这是能在 GPU 上反向传播的关键）；
- **解码器** p(x|z) 从隐变量重建 12 维属性（含 C00、C11）；
- **损失** = 重构 MSE + KL 散度（KL 把隐分布拉向 N(0,I)，保证隐空间连续、可插值）。

训练后，**解码器成为一个可微的"隐变量 → 渗透率张量"映射**，正是宏观 TO 所需的材料插值模型。

### 2.5 全局拓扑优化问题（建模宏观流场）

宏观器件离散为有限元网格。设计变量不是逐单元密度，而是**坐标神经网络**的权重：网络输入单元中心坐标 (x, y)（经 Fourier 特征映射），输出该单元的 (z₁, z₂, θ)。优化问题：

```
min_w   P(u)  = 耗散功率
s.t.    K(C(w)) u = f      (Stokes-Brinkman 有限元状态方程)
        g(w) ≤ 0           (接触面积/体积约束)
```

- 局部渗透率 C 由解码器从 (z₁,z₂) 得到，再按取向 θ 做张量旋转后填入全局刚度阵 K。
- 约束用**罚函数法**并随 epoch 逐步加重（continuation），避免一开始陷入不可行的尖锐局部最优。
- 状态方程的求解与对 w 的梯度，全部由 PyTorch 自动微分 + 可微稀疏求解器 `torch_sparse_solve` 完成。

> **坐标网络的作用**：相比逐单元独立变量，神经网络隐式正则化了设计场（相邻单元共享权重→空间平滑），并把设计维度与网格解耦，是近年 TO 的 neural reparameterization 思路。

---

## 3. 环境配置（WSL）

主机：Windows 11 + WSL2 **Ubuntu-22.04**；CPU 32 核、内存 62 GB、GPU NVIDIA RTX 5000 Ada (16 GB)。系统自带 Python 3.10.12。

由于 `python3-venv` 缺少 `ensurepip` 且无 sudo 权限，改用 **virtualenv** 建立隔离环境（放在 WSL 原生文件系统 `~/tomas-venv` 以获得最佳 I/O 性能）：

```bash
# 在 WSL Ubuntu-22.04 中执行
~/.local/bin/virtualenv ~/tomas-venv
source ~/tomas-venv/bin/activate
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.3 scipy==1.10.0 pandas==2.2.0 matplotlib==3.8.4 \
            pyyaml shapely joblib nbformat nbconvert tqdm
pip install torch_sparse_solve     # 原仓库 fluid_TO 依赖（未写入 requirements.txt）
pip install pypardiso              # 用于加速均匀化线性求解（见 §4）
```

要点：

- `torch 2.1.2+cu121`，`torch.cuda.is_available() == True`，识别到 RTX 5000 Ada。版本与原仓库 `requirements.txt`（numpy 1.26.3 / scipy 1.10.0 / pandas 2.2.0）一致。
- `torch_sparse_solve` 提供可微稀疏求解器，原仓库 `fluid_fe.py` 依赖它但未列入 `requirements.txt`；它有预编译 wheel，**无需 SuiteSparse / sudo**。
- 项目代码放在 Windows 侧 `E:\Working\reproduceTOMAS`（WSL 中为 `/mnt/e/Working/reproduceTOMAS`），便于在 Windows 查看；重计算为 CPU/GPU 密集型，`.mat` 文件 I/O 量不大，跨文件系统开销可接受。

### 对原仓库的少量修改（代码编写说明 · 让仓库可端到端运行）

复现中发现原仓库若干无法直接端到端运行的问题，做了最小化修正：

1. `dataset/supershape.py`：`import geopandas as gpd` 实际从未使用，且 geopandas 需要 GDAL。改为 `try/except` 守护，使模块在无 geopandas 时也能导入。
2. `fluid_TO/opt_constraints.py`：体积约束函数引用了未定义变量 `desired_fluid_vol_frac`（应为入参 `desired_vol_frac`），会在使用 VOLUME 约束（复现 Fig 12a）时报 `NameError`。已修正。
3. `vae/data_preprocess.py`：`stack_train_data` 中 `for normalization_type in normalization_type:` 误写（应为参数 `normalization_types`），导致 `UnboundLocalError`，VAE 数据归一化无法运行。已修正。
4. `vae/network.py`：编码器从 CPU 上的 `normal_dist.sample(...)` 采样，无法在 GPU 上训练。改为等价的 `torch.randn_like(mu)`（同样是 N(0,1) 采样，但跟随张量所在设备），使 VAE 可在 GPU 上训练（17000 epoch 由 CPU 约 61 min 缩短到 GPU 约 18 min）。
5. VAE 训练流程补充：原 `train_vae_main.ipynb` 没有保存归一化参数，但 `multiscale_TO_main.ipynb` 会加载 `vae/nomalization.pt`。在 `scripts/run_train_vae.py` 中补上保存 `{'max_feature','min_feature'}`。
6. `.mat` 数据形状：`scipy.io.savemat` 把 1 维数组存为 `(1, N)`，加载后与 `(N, …)` `hstack` 维度不匹配。在脚本中统一 `reshape(-1, 1)`。
7. `multiscale_TO_main.ipynb` 的训练循环里 `if method=='adam'`（枚举与字符串比较恒为假）及坐标张量 `requires_grad=True` 导致需要 `retain_graph`。改写为 `scripts/run_to.py`：坐标作为常量输入（`detach`），每个 epoch 重建计算图，逻辑更清晰。

---

## 4. MATLAB → Python 转换（核心代码编写工作）

### 4.1 转换对象

原仓库 `dataset/` 下的 MATLAB 文件：

| MATLAB 文件 | 作用 | Python 替代（本仓库 `dataset_py/`） |
|---|---|---|
| `fluidHomogenization.m` | 单个微结构的 Stokes/Brinkman 数值均匀化，返回 2×2 渗透率张量 | `fluid_homogenization.py` |
| `generate_homogenized_data.m` | 串行遍历所有微结构图像做均匀化 | `generate_homogenized_data.py`（**joblib 并行**） |
| `mstr_data_gen_main.m` | 驱动脚本（读/写 `.mat`） | 同上 + `run_data_generation.py` |

### 4.2 算法移植要点（`fluid_homogenization.py`）

`fluidHomogenization.m` 是 Andreassen & Andreasen (2014) 数值均匀化代码针对流体（Brinkman 罚项）的改写：4 节点双线性、压力稳定化单元，周期性边界，施加两个单位体力载荷（fx=1, fy=1），由单元平均速度得到渗透率对角分量。移植时的关键代码细节：

- **列主序对齐**：MATLAB 的 `reshape`/`(:)` 是列主序，Python(numpy) 默认行主序。所有展平/重排统一用 `order='F'`，否则单元-节点映射会错位。
- **1-based → 0-based**：周期映射阶段保留 1-based 节点/自由度编号以贴合原算术，仅在索引 Python 数组、构造稀疏矩阵时减 1。这样可逐行对照 MATLAB 源码，降低出错概率。
- **稀疏装配简化**：MATLAB 的 `sparse(iA,jA,sA)` 对重复 (i,j) 自动求和；`scipy.sparse.coo_matrix` 转 CSC 时同样求和。因此**只需保证三元组 (row,col,val) 正确**，不必逐一复刻原代码的 `kron`/转置索引技巧——直接用标准单元装配（速度块 8×8、速度-压力耦合 8×4 及其转置、压力稳定块 4×4）写出三元组即可，代码更短更可读。
- **固定压力基准**：求解时固定最后一个（压力）自由度为 0，对应 MATLAB 的 `A(1:end-1,1:end-1)\F(1:end-1,:)`，消除压力的常数自由度。

### 4.3 正确性验证（无 MATLAB 环境下的等价验证）

由于本机无 MATLAB / Octave，采用**物理解析解 + 对称性**验证移植正确性（`dataset_py/validate_homogenization.py`）：

| 检验 | 期望 | 结果 |
|---|---|---|
| Poiseuille 通道（沿 x，高度 h）C00 | ≈ h³/12 | 三个高度 h=0.3/0.5/0.7 的 C00/(h³/12) **恒为 1.018**（离散+有限 Brinkman 罚项导致约 1.8% 偏差） |
| 横向分量 C11 | ≈ 0（被堵） | 比 C00 小约 3 个数量级 |
| 转置检验（竖直通道） | C11↔C00 互换 | 完全镜像，x/y 记账正确 |
| 4 重对称方块 | C00=C11，C01≈0 | C00=C11，比值 abs(C01)/C00 ≈ 4e-15（机器零） |
| 随机 super-shape 数据集 | 非对角项远小于对角 | 比值 abs(c01)/c00 中位数 ≈ 1.1e-15（对称形状≈机器零），最大 ≈ 1.5e-2 |

结论：移植**物理正确**。

### 4.4 并行加速（代码编写说明 · 性能优化）

- **求解器加速**：scipy 的 SuperLU 在该 67500 自由度鞍点系统上单次求解约 **34.5 s**（比论文 MacBook M2 的 1.4 s 慢得多）。改用 **Intel MKL PARDISO**（`pypardiso`，自带 MKL、无需 sudo）后单次仅 **0.57 s**，**提速约 60×**，且与 SuperLU 结果一致到机器精度（`max|ΔCH| ≈ 4.5e-16`）。`fluid_homogenization.py` 优先用 PARDISO，缺失时回退 SuperLU。
- **样本级并行**：7000 个微结构的均匀化彼此独立，用 `joblib`（loky 后端）在 32 核上并行；每个 worker 设 `MKL_NUM_THREADS=1` 避免线程超额订阅（否则 32 进程 × 多线程会互相抢核），样本之间并行。
- **代码要点**：环境变量 `MKL_NUM_THREADS=1 / OMP_NUM_THREADS=1` 必须在 `import numpy` **之前**设置（见 `run_data_generation.py` 顶部），否则 MKL 已按多线程初始化。

**实测**：7000 个 super-shape（seed=27，无一被裁剪）在 32 核上：光栅化 64.9 s、并行均匀化 1041.9 s（约 17 min，等效 0.149 s/样本），总壁钟约 **18.5 min**（user 时间 502 min，≈27× 并行效率）。论文在 MacBook M2 上串行需 **164 min**，本复现壁钟加速约 **9×**。数据健康检查：7000 样本无 NaN/Inf，C00∈[2.9e-5, 0.376]、C11∈[3.7e-5, 0.376] 且全为正，非对角项中位数比对角项小约 15 个数量级，与论文"非对角项可忽略"一致。

---

## 5. 运行步骤

```bash
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS

# (1) 生成数据集（super-shape 采样 + 并行均匀化）：7000 样本
python -u dataset_py/run_data_generation.py --num-samples 7000 \
        --data-dir TOMAS/dataset --n-jobs -1

# (2) 训练 VAE（17000 epoch），保存 vae_net.pt + nomalization.pt
python -u scripts/run_train_vae.py

# (3) 隐空间实验：3.1 理想微结构(M*) / Fig 7 / Fig 10 / Table 1
python -u scripts/run_latent_space.py

# (4) TO 实验（示例）
python -u scripts/run_to.py --config TOMAS/notebooks/config_diffuser.yaml --tag diffuser
python -u scripts/run_to.py --config TOMAS/notebooks/config_bent_pipe.yaml --tag bent_perim
python -u scripts/run_to.py --config TOMAS/notebooks/config_bent_pipe.yaml \
        --constraint VOLUME --tag bent_vol
python -u scripts/run_to.py --config TOMAS/notebooks/config_bent_pipe.yaml \
        --fix-latent-file results/latent/M_star.npy --tag bent_orient
python -u scripts/run_to.py --config TOMAS/notebooks/config_biffurcated_pipe.yaml \
        --desired-perim 70 --tag bifurcated

# (5) Pareto 前沿（Fig 14）+ 真实-FEA 验证
python -u scripts/run_pareto.py --perims 40,50,60,70,80
python -u scripts/run_validate_design.py --config TOMAS/notebooks/config_bent_pipe.yaml \
        --design results/to/bent_perim/design.npz
```

**一键复现全部实验**（VAE 训练完成后）：

```bash
bash scripts/run_all_experiments.sh        # §3.1–§3.7 全部 + 验证，日志见 results/logs/
python scripts/collect_results.py          # 汇总所有 metrics.json 为对照表
```

### 计算成本（§3.6，本机实测）

| 步骤 | 论文 (MacBook M2) | 本复现 |
|---|---|---|
| 数据生成 7000 样本（Algorithm 1） | 164 min（串行） | **18.5 min**（32 核并行，≈9×） |
| VAE 训练 17000 epoch（Algorithm 2） | 90 min | **≈18 min**（GPU） |
| 弯管 TO 20×60 × 301 epoch | 32 min | **≈6 min** |
| 扩散器 TO 15×15 × 301 epoch | 1.5 min | **≈0.5 min** |

---

## 6. 复现结果与论文对比

### 6.1 数据集与 VAE

- 数据集：7000 个 super-shape（150×150 均匀化），并行生成耗时 18.5 min。
- VAE：12→2→12，600 隐藏单元，17000 epoch（GPU 约 18 min），重构损失（归一化 MSE）由 8.2e-2 收敛到约 9.8e-3。
- §3.1 理想微结构 M*（vf≈0.25、在物理自洽候选中取**真实**(重新均匀化) trace(C) 最大）：本复现选得隐坐标 z=(-0.26, 0.29) 的微结构，解码 vf=0.250、C00=0.0107、C11=0.0127；其重新均匀化的真实 vf=0.20、真实周长=0.94、真实 trace(C)=0.036，解码器在该点自洽（M* 选取的修正过程见 §6.5）。

> 说明：本复现 VAE 的逐样本重构精度（Table 1 量级）整体略逊于论文，主要因 2 维隐空间对 12 维属性是强压缩、且数据集为独立随机生成。但**下游 TO 结果与论文吻合良好**，说明可微隐空间作为设计空间是有效的。

### 6.2 拓扑优化结果对比

| 实验 | 量 | 论文 | 本复现 | 说明 |
|---|---|---|---|---|
| §3.5/Fig15 扩散器(接触面积60) | 最大速度幅值 | 2.81（Ansys 2.87） | **2.77** | 误差约 1.4%，吻合优秀 |
| §3.4 扩散器 | 接触面积/功率 | 60 / — | 60.05 / 28.35 | 约束满足 |
| §3.2/Fig11 弯管(仅取向) | 耗散功率(真实-FEA) | 15.1 | 解码 28.88 / **真实 15.86** | 修正 M* 后吻合(差约 5%)，见 §6.5 |
| §3.3b/Fig12b 弯管(接触面积75.69) | 耗散功率(解码) | 7.56 | 6.58 | 需对齐真实接触面积，见 §6.5 |
| §3.3a/Fig12a 弯管(体积0.75) | 耗散功率 | 9.61 | 7.99 | 真实接触面积差异大，不可直接比，见 §6.5 |
| §3.3b 验证(真实FEA,对齐接触面积) | 真实功率 @ 真实接触面积≈78.5 | 7.87 | 6.93 | 对齐口径后基本相当(低约 12%) |
| §3.7/Fig16 分叉管 | 设计(接触面积70) | 见图 | ca=69.4, 功率 21.77 | 约束满足，设计合理 |

各 TO 实验的最大速度幅值在 Pareto 五个点上稳定在 2.76–2.79，与论文 2.81 一致。

### 6.3 讨论：吻合点与差异

**高度吻合**：(1) 扩散器最大速度 2.77 vs 2.81（1.4%）；(2) 弯管接触面积约束设计的真实-FEA 验证误差（功率 2.5%、接触面积 5.6%）与论文（4.0%、3.7%）同量级；(3) 所有约束均被满足，设计在物理上合理。

**存在差异（经二次核查与修正）**：
- §3.2 仅取向弯管：首轮 7.13 与论文 15.1 差约 2×，**已查明是 stale-file bug**（`run_latent_space.py` 在 VAE 回退 12 维后仍以 `input_dim=10` 加载，磁盘 `M_star.npy` 是旧 10 维退化产物）。**修复后真实-FEA 功率 15.86，与论文 15.1 吻合（差约 5%）**，见 §6.5。
- §3.5 Pareto 前沿在首轮复现中**非单调**（论文为单调增），源自本 VAE 隐空间形貌与梯度优化的局部最优；已用暖启动连续化修复。

### 6.4 改进尝试与二次复现

**有效改进（已采用）：**
- **Pareto 暖启动连续化**（`scripts/run_pareto.py`）：按接触面积升序扫描，每个点用上一点收敛的网络权重热启动（同伦/连续化）。**前沿由非单调变为单调**：接触面积 50→80，真实-FEA 功率由 28.3 单调升至 31.1，与论文 Fig 14 定性一致。
- **多随机种子 best-of**（`scripts/run_best_of.py`）：headline 算例用 seeds {77,1,2} 各跑一次，取满足约束且功率最低者并经真实-FEA 验证，降低对局部最优的敏感性。

**无效甚至有害的尝试（已回退，记录以供参考）：**
- **按论文文字改 10 维输入**（去掉近常数 cx,cy）：重构损失更低，但 TO **更差**（扩散器 ~35 vs 12 维的 ~28）。**更低的重构损失 ≠ 更好的 TO 设计**（隐空间几何更关键）。已保留 12 维。
- **过滤退化样本**（去掉 vf<0.02 的近空胞元）：这些近空胞元恰是**渗透率最高**的微结构，是耗散最小化 TO 构造低阻流道所必需；过滤后功率全面升高。已回退。
- **余弦 LR + 增至 30000 epoch**：重构损失更低，但 TO 无改善。最终采用忠实论文的平坦 LR / 17000 epoch。

**改进后对比（解码器→真实-FEA）：**

| 实验 | 论文 | 本复现（解码→真实） | 评价 |
|---|---|---|---|
| §3.7 分叉管(接触面积70) 功率 | 见图 | 21.77→21.57 | 约束满足，验证误差<1% |
| §3.5 Pareto 单调性 | 单调增 | **单调**（暖启动修复）| 已修复 |
| §3.4 扩散器最大速度 | 2.81 | 2.77 | 吻合(1.4%) |
| §3.4 扩散器功率(接触面积60) | ~22 | 28.4（最优可行解）| 偏高，VAE 设计空间所致 |

**结论**：Pareto 前沿经暖启动**恢复单调**；扩散器最大速度吻合(1.4%)。扩散器绝对功率（~28 vs ~22）仍偏高，源自本独立训练 2 维隐空间 VAE 的设计空间本身（系统实验确认无法靠调重构精度改善）。弯管类（§3.2/§3.3）的对比需对齐口径，详见 §6.5。

## 6.5 公平对比与诚实归因（弯管，三次复现）

> 起因：将本复现弯管功率与论文 Fig 12 对照时发现"本复现功率更低"。深入核查后确认，**"功率更低"主要源自"对比口径不一致 + 一处 stale-file bug"，而非真的更优**。以下据真实-FEA 重新归因，并据此**更正此前"优于论文"的表述**。

**(1) §3.2 仅取向：stale-file bug 修复后与论文吻合。** `run_latent_space.py` 在 VAE 回退 12 维后仍写死 `input_dim=10`，磁盘 `M_star.npy` 是旧 10 维退化产物（真实接触面积≈0 的空胞）。修复 `input_dim=12`、并把 M* 选取改为"在物理自洽候选中取**真实**(重新均匀化) trace(C) 最大"后，新 M* 落在隐空间中心（z=(-0.26,0.29)，训练充分区，解码器自洽）。重跑 §3.2：**真实-FEA 功率 15.86，与论文 15.1 吻合（差约 5%）**（解码 28.88，该点解码器误差 82%，反映 2 维 VAE 重构精度有限）。

**(2) §3.3b 接触面积约束：对齐"真实接触面积"后，本复现与论文相当（非大幅更优）。** 约束施加在**解码周长**上，而本复现解码器**高估**接触面积（论文相反，低估），故同一名义目标下本复现设计真实接触面积偏小→实心阻挡更少→功率自然更低。把解码目标上调使真实接触面积命中论文量级：

| 解码目标 | 解码功率 | 真实功率 | 真实接触面积 |
|---|---|---|---|
| 75.69（首轮）| 6.58 | 6.42 | 72.59 |
| 79 | 7.03 | 7.11 | 75.30 |
| 82 | 6.99 | **6.93** | **78.43** |
| 论文 Fig 12b | 7.56 | 7.87 | 78.49 |

在论文真实接触面积 78.49 上对齐，本复现真实功率 **6.93 vs 论文 7.87**——仅低约 12%，且真实功率随真实接触面积**非单调**（局部最优 ±0.5 抖动）。首轮"6.42"看似更优，实因它工作在更小的真实接触面积(72.59)上。

**(3) §3.3a 体积约束：接触面积实现量差异大，不宜直接比。** 同样约束 vf=0.75，但论文设计接触面积 75.69，本复现真实接触面积仅 60.9（更大开孔→耗散更低），绝对功率（7.89 vs 9.61）非同口径。

**总结论（更正措辞）**：此前"弯管类优于论文"**不成立**；对齐口径后本复现弯管结果与论文**基本相当**（§3.2 真实 15.86 vs 15.1；§3.3b 真实 6.93 vs 7.87 @ 同接触面积）。本复现价值在于端到端打通 + 诚实的真实-FEA 归因与 bug 修复。

---

## 7. 目录结构

```
reproduceTOMAS/
├── TOMAS/                     # 原仓库（含 §3 所述 7 处小修改）
├── dataset_py/                # MATLAB→Python 的均匀化与并行数据生成
│   ├── fluid_homogenization.py        # fluidHomogenization.m 的 Python 移植（核心）
│   ├── generate_homogenized_data.py   # 并行替代
│   ├── run_data_generation.py         # 端到端并行数据生成（采样+均匀化）
│   └── validate_homogenization.py     # 物理解析解验证
├── scripts/                   # 复现驱动脚本
│   ├── run_train_vae.py               # VAE 训练（GPU）
│   ├── run_latent_space.py            # §3.1 M* / Fig7 / Fig10 / Table1
│   ├── run_to.py                      # 通用 TO 驱动（§3.2–3.4, 3.7）
│   ├── run_pareto.py                  # §3.5 Pareto 扫描(暖启动连续化) + Fig14
│   ├── run_best_of.py                 # 多随机种子 best-of
│   ├── run_validate_design.py         # 真实-FEA 验证
│   ├── collect_results.py             # 汇总所有指标
│   └── run_all_experiments.sh         # 一键复现 §3.1–§3.7
├── results/                   # 生成的图像与指标
└── docs/复现文档.md            # 完整复现文档
```
