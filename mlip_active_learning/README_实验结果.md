# 实验结果完整说明

## 目录结构

```
mlip_active_learning/
├── README_实验结果.md          ← 本文件
├── results_schnet_ms25/        SchNet + MS25 主动学习 (36个CSV)
├── results_mace_al/            MACE 微调主动学习 (18个CSV)
├── results_metrics/            5项评测指标 (1个CSV)
├── results_rho/                ρ分布统计 (待补充)
└── experiment_plan.md          原始实验计划
```

---

## 一、results_schnet_ms25/ — SchNet + MS25 主动学习

**36个CSV文件**：6体系 × 3种子(42/52/62) × 2版本

### 8策略版 (ms25_{体系}_seed{42,52,62}.csv, 18个)
列名：A_random, B_gmm_uncertainty, C_ensemble_qbc, D_mc_dropout, E_diversity, F_latent_clustering, G_hybrid_weighted, H_hybrid_twostage

### 10策略版 (ms25_9strat_{体系}_seed{42,52,62}.csv, 18个)
列名：以上8列 + I_aud_rank + J_aud_batch + K_aud_bald + L_rho_diagnostic

每行 = 一轮主动学习迭代 (N=50, 65, 80, 95, 110, 125, 140)

### 体系缩写
| 文件名 | 全称 | 类型 |
|--------|------|------|
| FeNiCrCoCu_HEA | FeNiCrCoCu 高熵合金 | 化学无序 |
| MgO_surface | MgO(100)表面 | 离子晶体表面 |
| Pt_CH_activation | Pt(111) C-H断键 | 催化金属表面 |
| Zr_oxide_amorphous | 非晶ZrO₂ | 非晶氧化物 |
| liquid_water | 液态水 | 分子液体 |
| zeolite | 沸石骨架 | 多孔骨架 |

---

## 二、results_mace_al/ — MACE 微调主动学习

**18个CSV文件**：6体系 × 3种子

列名：A_random, C_uncertainty, E_diversity, G_hybrid_weighted, I_aud_rank, J_aud_batch, K_aud_bald, L_rho_diagnostic

每行 = 一轮主动学习 + MACE微调迭代 (N=50, 65, 80, 95, 110, 125, 140)

---

## 三、results_metrics/ — 5项评测指标

`evaluation_metrics.csv` — 6个体系在MACE零样本下的评测：

| 指标 | 含义 |
|------|------|
| md_stable | 300K NVT MD是否稳定 (200步, 0.5fs) |
| md_max_force | MD中最大力值 (eV/Å) |
| ev_plausible | E-V曲线是否物理合理 |
| ev_violations | E-V单调性违规次数 |
| soap_coverage | 结构空间覆盖度 |

---

## 四、跨体系核心结论

### SchNet版 (8策略)
- 最佳：E_diversity +12.9% (5/6胜出)
- G_hybrid_weighted +3.9% (3/6)

### SchNet版 (10策略, 含I/J/K/L)
- 最佳：I_aud_rank +4.1% (沸石+37%)
- L_rho_diagnostic 4/6胜出 (最稳健)

### MACE微调版 (8策略)
- 最佳：G_hybrid_weighted +27% (排除ZrO₂)
- L_rho_diagnostic +9.7% (4/5胜出)
- I_aud_rank 在沸石+48%

### ρ分布统计
- ρ均值 ≈ +0.05 (U/D通常不相关)
- ρ < -0.3 仅17%触发自适应
- 沸石ρ波动最大 (-0.64到+0.65)，自适应最有效

---

## 五、策略说明

| 编号 | 策略 | 原理 |
|------|------|------|
| A | Random | 随机选择 |
| B | GMM Uncertainty | 聚类距离 |
| C | Ensemble QBC | 委员会方差 |
| D | MC-Dropout | 多前传方差 |
| E | Diversity | 嵌入FPS |
| F | Latent Clustering | k-means聚类 |
| G | Hybrid-Weighted | α=0.5 加权U+D |
| H | Hybrid-TwoStage | 不确定性筛→多样性选 |
| I | AUD-Rank | 自适应α (ρ驱动) |
| J | AUD-Batch | 批选择(贪心U+minDist) |
| K | AUD-Bald | BALD信息增益+自适应α |
| L | ρ-Diagnostic | ρ<-0.3触发自适应，否则α=0.5 |
