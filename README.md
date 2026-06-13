# 复现 TOMAS（多尺度流体拓扑优化 + VAE + super-shapes）

复现论文 *TOMAS: topology optimization of multiscale fluid flow devices using
variational auto-encoders and super-shapes*（Padhy, Suresh, Chandrasekhar，
Struct Multidiscip Optim 2024, 67:119），原始代码：https://github.com/UW-ERSL/TOMAS

**完整复现过程见 [`docs/复现文档.md`](docs/复现文档.md)。**

## 本工作做了什么
- 在 **WSL (Ubuntu-22.04)** 中端到端配置并运行了全部代码（数据生成 → VAE 训练 →
  多尺度流动拓扑优化 → 真实 FEA 验证），复现论文 §3.1–§3.7 的全部数值实验与图像。
- 把原仓库中 **MATLAB 部分（流体数值均匀化）完整改写为 Python**
  （`dataset_py/fluid_homogenization.py`），并以解析解验证其正确性。
- 用 **MKL PARDISO（单次提速约 60×）+ 32 核 joblib 样本级并行** 加速样本生成：
  7000 个微结构均匀化由论文的 164 min（串行）降到 **18.5 min**（约 9× 壁钟加速）。
- 修复了原仓库 7 处导致无法端到端运行的问题（见文档 §2）。

## 快速开始（WSL 内）
```bash
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
python -u dataset_py/run_data_generation.py --num-samples 7000 --data-dir TOMAS/dataset
python -u scripts/run_train_vae.py
bash scripts/run_all_experiments.sh
python scripts/collect_results.py
```

## 关键结果（与论文对照，详见文档 §5）
| 指标 | 论文 | 本复现 |
|---|---|---|
| 扩散器最大速度幅值（接触面积 60） | 2.81 | **2.77**（1.4%） |
| 弯管真实-FEA 验证误差（功率 / 接触面积） | 4.0% / 3.7% | 2.5% / 5.6% |
| 数据生成 7000 样本 | 164 min | **18.5 min** |

速度场、约束满足、真实-FEA 验证误差等关键指标与论文吻合；少数定量差异
（Pareto 单调性、个别耗散功率绝对值）源自独立训练的 2 维隐空间 VAE 重构精度有限
与梯度优化局部最优，属从零复现 VAE 类方法的预期范围。

---

## 本分支：方法 A — 有效接触面积度量（p8a-effective-area）

**动机**：从流体力学看，星形微结构不利于流动（多凹腔=死区回流、各向同性高阻），其凹腔周长
"几何上湿润、流动上无效"，却被几何周长式的接触面积约束当成廉价周长来源。本分支把接触面积约束
里的几何周长换成**有效周长 = 几何周长 × 凸性折扣** `conv = clamp(area/perim² / 0.3136, 0, 1)`
（数据标定：圆/叶 conv≈0.97 满分，星形 conv≈0.17 打 6 折），从源头消除星形的"作弊周长"。

**用法**：`scripts/run_to.py --effective-area --kappa-ref 0.3136`（见 `scripts/run_p8a.sh`）。

**结果图（P6b VAE + 有效面积约束，真实-FEA）**：

| 弯管 Fig 11 (真实 13.06) | 分叉管 Fig 16 (29.7 / CA 54) | 扩散器 Fig 13 (52.6 / CA 46) |
|:---:|:---:|:---:|
| <img src="results/to/bent_orient_p8a/design.png" width="95"> | <img src="results/to/bifurcated_p8a/design.png" width="160"> | <img src="results/to/diffuser_p8a/design.png" width="160"> |

扩散器视觉最佳（几乎全椭圆/卵形/透镜密铺，贴论文 Fig 13b），直接证实"奖励有效表面 → 自发好流动形状"；
代价是接触面积顶不满（好流动形状周长本就少）。三方法横向对比见 `pure-claude` 分支 README。
