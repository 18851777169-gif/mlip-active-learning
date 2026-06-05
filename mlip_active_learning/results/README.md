# 实验结果说明

## 主实验结果（8策略 SchNet + MS25）

`ms25_{体系}_seed{42,52,62}.csv` (18个文件)

SchNet 模型（轻量图神经网络）学习 MACE 标注的能量，8 种采集策略 × 3 个随机种子的主动学习对比。

每个 CSV 的列名即策略：
- A_random: 随机选择（基线）
- B_gmm_uncertainty: GMM 不确定性（聚类距离）
- C_ensemble_qbc: 委员会查询（2 模型 ensemble 方差）
- D_mc_dropout: MC-Dropout（多次前传方差）
- E_diversity: 多样性（嵌入空间 FPS）
- F_latent_clustering: 潜在空间聚类
- G_hybrid_weighted: 混合加权（提出方法，α=0.5）
- H_hybrid_twostage: 混合两阶段（提出方法，top 30% → FPS）

每行对应一轮主动学习迭代（N=50, 65, 80, ... 170）

体系缩写：
- FeNiCrCoCu_HEA: 高熵合金
- MgO_surface: MgO(100)表面
- Pt_CH_activation: Pt(111) C-H 活化
- Zr_oxide_amorphous: 非晶 ZrO2
- liquid_water: 液态水
- zeolite: 沸石骨架

## MACE 微调实验

`mace_al_{体系}_seed{42,52,62}.csv` (18个文件)

MACE-MP-0 预训练模型微调，4 种采集策略 × 3 种子。

列名：
- A_random: 随机选择
- C_uncertainty: Ensemble 不确定性
- E_diversity: MACE 嵌入多样性
- G_hybrid_weighted: 混合加权

## 评测指标

`evaluation_metrics.csv`

6 个材料体系的 5 项评测指标：
- md_stable: MD 稳定性（300K NVT，力是否爆炸）
- md_max_force: MD 中最大力值 (eV/Å)
- ev_plausible: E-V 曲线物理合理性
- ev_violations: E-V 单调性违规次数
- soap_coverage: 结构空间覆盖度

## 快速实验（前期验证）

`fast_experiment_*.csv`: Cu LJ 团簇快速验证（早期开发阶段）
`aggregate_3seeds_mean.csv`: LJ 实验 3 种子均值

## 原始数据

`../data/ms25_labeled/{体系}.pkl` (6个文件)

MACE-MP-0 标注的 ASE Atoms 结构，每个包含：
- 原子坐标、种类、晶胞
- info["energy"]: MACE 计算的总能量 (eV)
- info["forces"]: MACE 计算的原子力 (eV/Å)
- info["system"]: 体系名称
