# VRSE Phase 3：C-MAPSS 高维区域授权最小验证计划

> 状态：**FROZEN / PHASE-3B PASS**  
> 创建日期：2026-07-28  
> 前置快照：`results/PHASE2B_SNAPSHOT.md`  
> Phase-3A 因 ID-fit-only 标准化产生最高约 `1.76e8` 的 z-score，机械
> `PIVOT_CORE` 不构成机制证据。Phase-3B 按唯一修订首次干净执行并得到 `PASS`；
> 本实验管线不再调整。冻结结果与哈希见 `results/PHASE3B_SNAPSHOT.md`，复现步骤见
> `docs/Phase3_Runbook.md`。

## 1. 唯一研究问题

Phase 3 只回答一个问题：

> 当输入从 1D 标量扩展到真实任务中的多传感器向量时，VRSE 能否在不改变冻结
> ID 服务、也不连带开放另一个未知工况的前提下，验证并局部授权一个已经学会稳定
> 新工况的影子残差专家，同时拒绝发生概念反转的候选？

本阶段不是 RUL 精度竞赛，不宣称生产安全，不比较完整持续学习 SOTA，也不同时扩展
分类、多输出、序列网络、GPU、延迟标签或插件系统。

## 2. 目标任务与选择理由

### 2.1 数据集

选择 **NASA C-MAPSS FD002**。官方说明将 C-MAPSS 描述为多条发动机退化
多变量时间序列；FD002 含 260 条训练轨迹、259 条测试轨迹、六种运行条件和一种
故障模式。每个周期包含 3 个运行设置和 21 个传感器测量。

- 官方入口：<https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- NASA Science 数据许可说明：<https://science.data.nasa.gov/about/license>
- 原始任务引用：A. Saxena, K. Goebel, D. Simon, N. Eklund, PHM 2008

必须准确表述它的性质：这是面向真实预测性维护问题的**工业仿真 benchmark**，
不是实际飞机现场采集数据。NASA 数据页当前把本条目的许可证显示为
“License not specified”；因此仓库首版只提供官方下载说明、校验哈希和预处理脚本，
不重新分发原始压缩包。Phase 0 必须再次核对下载页及附带文件是否有更具体条款。

### 2.2 为什么它是最小且有代表性的选择

- 与当前库保持同一学习范式：有监督流式**标量回归**；
- 单周期输入自然得到 24 维向量，无需引入图像或序列模型才能达到高维；
- 六种运行工况提供真实的输入分布结构，可定义 ID、新工况与未授权工况；
- 训练轨迹运行至失效，可为每个周期离线构造 RUL 标签并严格按发动机划分数据角色；
- 数据规模足以做五个 seed，又不需要大规模算力。

不选择图像分类，是因为那会同时改变输入结构、输出类型、损失和影子学习器，失败后
无法判断是高维支持集还是分类扩展造成。不选择更大的 N-CMAPSS，是因为其存储与
预处理成本对本轮机制验证没有必要。

## 3. 冻结的任务构造

### 3.1 输入与目标

- 输入 `x ∈ R^24`：3 个运行设置 + 21 个传感器值；
- 不把 unit id 和 cycle index 输入模型；
- 目标：`y = min(125, max_cycle(unit) - cycle)`；
- 所有模型只做单步标量 RUL 回归，不使用滑动窗口；
- **Phase-3A 冻结选择（已失效）**：每个 seed 只用 ID-fit 估计尺度；
- **Phase-3B 修订**：在任何监督训练前，仅用已隔离的 `regime_discovery` units 1–20
  的全部无标签 24 维输入计算一套跨工况公共均值与标准差，五个 seed 共享；
- 标准差下限仍为 `1e-6`。任何落到下限的特征若在完整原始 FD002 输入中仍有变化，
  P0 立即失败。不得用 shadow、promotion-validation、post-test 或其标签重估尺度。

单周期模型不是为取得最优 RUL 分数，而是为了只引入一个新变量：高维支持几何。

### 3.2 运行工况的机械选择

FD002 不直接给出离散工况标签，只有三个运行设置。工况发现不得使用 RUL 标签：

1. 固定 unit 1–20 为 `regime_discovery`，只读取三项运行设置；这些 unit 不进入任何
   训练、校准或测试角色；
2. 在该输入上标准化后运行 `KMeans(k=6, random_state=31415, n_init=20)`；
3. 将六个中心按原始三维坐标字典序重编号，消除 KMeans 标签置换；
4. `C_ID` = discovery 数据中样本数最多的工况；若并列取字典序较小者；
5. 按标准化中心到 `C_ID` 中心的欧氏距离排序：最远者为稳定新工况 `C_NEW`，
   次远者为从未授权的 `C_UNKNOWN`；若并列仍取字典序较小者。

选择结果和中心必须在任何标签模型训练之前写入 `results/phase3_data_manifest.json`。
不能因为某个工况较难学习而事后改选。

### 3.3 按发动机隔离数据角色

其余 240 个 unit 对每个 seed 独立随机排列。固定 seeds：

```text
4300, 4301, 4302, 4303, 4304
```

依次分配：

| 角色 | unit 数 | 使用的工况 | 是否使用标签 |
|---|---:|---|---|
| ID fit | 60 | `C_ID` | 是，训练 baseline/fit-time GP |
| ID calibration | 30 | `C_ID` | 否，只校准 deploy uncertainty tau |
| ID guard | 30 | `C_ID` | 是，只评估旧服务不变量 |
| Shadow observe | 50 | `C_NEW` | 是，只更新隔离候选 |
| Promotion validation | 35 | `C_NEW` | 是，只用于一次晋升考试 |
| Post-decision test | 35 | `C_NEW` 与 `C_UNKNOWN` | 是，只用于最终评估 |

同一个 unit 不得跨角色。每个角色保留其 unit 内原始 cycle 顺序；跨 unit 按
`(cycle, unit_id)` 排序形成确定性的 fleet stream。官方 test trajectories 不使用，
因为本阶段需要每个周期的完整 RUL 标签。

### 3.4 两条配对流

每个 seed 在完全相同的 `x`、unit 划分和预算上运行两条流：

1. `stable_condition`：observe、validation 和 post-test 全部使用真实 capped RUL；
2. `reversed_condition`：observe 阶段仍使用真实 capped RUL；从 promotion validation
   开始，标签机械替换为 `125 - y`。候选所学规律在考试前失效。

第二条是明确标注的半合成负对照，只用于检验“样本够多但能力已失效时是否仍会
晋升”，不作为 C-MAPSS 的自然现象或现实发生率证据。

## 4. Phase 3 唯一新增机制：高维支持集

Phase 2B 的生命周期和 GP 后验保持不变。Phase 3 只把 1D observed span 替换为一个
内部固定实现 `KNNFeatureRegion`，不把它提升为公共插件协议。

### 4.1 特征空间

- `PhiSN` 接收 24 维输入，输出 32 维冻结特征；
- 输入投影和后续线性层都施加谱归一化；
- 对每个谱归一化层记录最大奇异值，任何层超过配置上限 `0.95 + 1e-3` 即前置失败；
- RFF 维数固定 128；中位长度尺度最多在 2048 个 ID embedding 上确定性估计，避免
  构造全数据 `N×N` 距离矩阵。

### 4.2 区域定义

从 shadow-observe embedding 中用确定性 farthest-point sampling 选至多 512 个原型：
首点取离候选 embedding 均值最近的样本，后续每次取离已有原型集合最远的样本。

对任意 embedding `z` 定义：

```text
d5(z) = z 到 512 个原型的第 5 近欧氏距离
```

- 距离半径 `r_support`：在 promotion-validation 输入的 `d5` 上使用与 Phase 2B 相同的
  单侧 95% 内容 / 95% 置信 tolerance limit；
- 候选不确定性上限 `tau_region`：在同一 validation 输入的 shadow GP uncertainty
  上使用相同 tolerance limit；
- 授权条件：`d5(z) <= r_support AND u_shadow(z) <= tau_region`；
- validation 样本少到 tolerance limit 不可计算时 fail closed；
- 若任何 ID-guard 样本满足授权条件，cond4 失败，不得晋升。

validation 的输入可以参与支持半径校准，但其标签只参与能力考试；post-test 与
`C_UNKNOWN` 从不参与区域构造或阈值确定。

### 4.3 路由语义

```text
未晋升                    -> frozen baseline
已晋升且在 KNNFeatureRegion -> frozen promoted GP residual expert
已晋升但不在区域            -> frozen baseline, exact fallback
```

高维区域必须存储在 proposal/deployment snapshot 中，晋升后继续 observe 不得改变
已服务的原型、半径、tau 或 GP 后验。

## 5. 固定模型与预算

### 5.1 冻结 baseline

```text
MLP: 24 -> 64 -> 64 -> 1
activation: ReLU
optimizer: Adam(lr=1e-3, weight_decay=1e-4)
batch size: 256
epochs: 100 fixed, no early stopping
loss: MSE
```

baseline 每个 seed 只在 ID-fit 上训练一次，然后被五种方法共享并冻结。不得针对
`C_NEW`、promotion validation 或 post-test 调整网络或训练轮数。

### 5.2 VRSE 固定参数

- `hidden_dim=32`, `n_blocks=2`, `sn_multiplier=0.95`；
- `rff_dim=128`, `prior_std=1.0`；
- `tau_percentile=95`, `tau_confidence=0.95`；
- `promotion_rmse_ratio=0.80`, `promotion_q95_ratio=1.00`；
- 一次 scheduled exam：完整观察 50 个 shadow unit 后执行；
- CPU 为本阶段规范设备，不把 GPU 支持作为 Phase 3 成败条件。

`noise_std` 不沿用 1D 的绝对 `0.05`。为消除 RUL 单位尺度影响，在每个 seed 的
ID-fit residual 上机械设为：

```text
noise_std = max(1e-3, 0.10 * std(ID-fit residual))
```

这是协议公式，不允许用 validation/test 调整。

## 6. 五种方法

所有方法共享数据、baseline、PhiSN、RFF 映射、候选更新顺序和计算预算。

| 方法 | 作用 |
|---|---|
| Frozen | baseline 永不适应，给出安全/性能参考 |
| Online-ungated | 同一 GP 残差候选边观察边全局服务，代表无隔离的最大塑性 |
| Static-reject | 只用 fit-time SNGP uncertainty 门拒绝，不允许候选晋升 |
| Shadow-global | 候选隔离学习并参加相同 cond1/cond2 考试，通过后全局替换残差专家 |
| VRSE-KNN | 候选隔离学习、完整考试、只在高维支持区域内获得权限 |

`Shadow-global` 与 `VRSE-KNN` 的候选能力判据必须相同；二者唯一实质差异是全局授权
还是区域授权。不得加入另一套更强模型使比较失去机制可解释性。

总矩阵：2 streams × 5 methods × 5 seeds = **50 runs**。baseline/PhiSN checkpoint
在同一 seed 的十个 run 间复用，不重复训练。

## 7. 前置硬门

以下检查在正式矩阵前运行。失败后不得通过换工况、换数据集或调 promotion 阈值救援。

### P0 数据与实现完整性

- 官方压缩包 URL、下载日期、SHA-256、文件行数写入 manifest；
- 六个工况均非空；六类 unit 角色无交集；
- 标准化统计量只来自冻结的无标签 `regime_discovery` units 1–20；所有 seed 完全一致；
- 归一化审计必须证明没有“标准差落到 `1e-6` 下限但在 discovery 外变化”的特征，
  并报告全部原始行上的最大绝对 z-score；
- Phase 2B 的 24 项测试与两个 Stage-4C 回归锚点继续通过；
- 高维 batch/incremental GP 在一个小 fixture 上误差 `<1e-10`（float64 后验）；
- region snapshot 在晋升后继续 observe 时逐位不变。

### P1 fallback 有意义

在 ID guard 上，baseline RMSE 必须比“恒预测 ID-fit 标签均值”低至少 20%，且在
至少 4/5 seeds 成立。否则 `STOP_TASK`：没有一个值得保护的旧服务。

### P2 新工况最初确为未知

在任何 shadow update 前，fit-time SNGP 门必须在 `C_NEW` promotion-validation 上
拒绝至少 90% 样本，至少 4/5 seeds 成立；同时 ID-calibration 的经验接受率报告但
不作为二次调阈值依据。失败判为 `PIVOT_DETECTOR`。

### P3 候选具备可晋升能力

稳定流最终 shadow 在独立 validation 上必须同时满足 cond1/cond2，至少 4/5 seeds。
失败判为 `PIVOT_LEARNER`；这说明当前 GP 容量/特征不足，不能把失败归咎于区域构造。

## 8. 主要指标与成功标准

### H1 隔离与初始回退

在两条流、所有 seed、每次 shadow 更新检查点：

- VRSE 晋升前输出与 frozen baseline 的 `max_abs_diff < 1e-6`；
- shadow 更新不得改变当前部署输出。

要求 **10/10 seed-stream 通过**。

### H2 决策正确性

- `stable_condition` 晋升：至少 4/5 seeds；
- `reversed_condition` 错误晋升：至多 1/5 seeds。

### H3 区域隔离

对所有成功晋升的稳定 seed：

- ID guard 晋升路由率 `== 0`，预测最大变化 `<1e-6`；
- `C_UNKNOWN` post-test 晋升路由率 `== 0`，输出与 baseline 最大差 `<1e-6`；
- `C_NEW` post-test 晋升路由覆盖率 `>= 0.80`。

H3 按每个 seed 逐项报告，要求至少 4/5 seeds 全部通过；不能只看跨 seed 平均。

### H4 稳定新工况的实际收益

在独立 `C_NEW` post-test 上，至少 4/5 seeds 同时满足：

- VRSE RMSE 相对 Frozen 改善至少 10%；
- VRSE RMSE 不超过 Shadow-global 的 1.20 倍；
- VRSE 路由到专家的样本上，RMSE 相对 Frozen 改善至少 20%。

同时报告 RMSE、MAE、绝对误差 q95、NASA asymmetric score，但只有上述三条参与
机械判决。

### 次要工程指标

报告 fit 时间、每千次 observe 时间、单样本推理延迟、峰值内存、proposal/region
序列化大小。这些数据用于开源 README，不设 PASS/FAIL 门槛。

## 9. 机械判决

按以下顺序执行，先满足者即为最终判决：

1. 数据损坏、角色泄漏、测试/回归门失败：`INVALID`；修 bug 后保留原记录并在重跑前
   写 amendment；
2. P1 失败：`STOP_TASK`；该任务没有形成有意义的安全 fallback；
3. P2 失败：`PIVOT_DETECTOR`；高维 SNGP 未形成所需未知性分离；
4. P3 失败：`PIVOT_LEARNER`；候选容量不足，区域授权尚未得到公平检验；
5. H1 或 H2 失败：`PIVOT_CORE`；隔离或独立验证的核心语义未推广成功；
6. H1/H2 通过，但 H3 或 H4 失败：`CONDITIONAL_PIVOT_SUPPORT`；核心隔离和能力考试
   成立，但高维支持覆盖/隔离未形成可用折中；
7. P0–P3 与 H1–H4 全部通过：`PASS`。

禁止在看到 Phase 3 结果后：改工况选择、改 seed、删困难 unit、改变 RUL cap、调整
KNN 的 `k`/原型数/容忍限、放宽 4/5 规则或把半合成反转流改成更容易的负对照。

## 10. 最短执行顺序与交付物

### T0 数据预检与冻结

实现官方下载/校验、解析、RUL、工况发现和 unit 角色划分；输出
`results/phase3_data_manifest.json`。只做输入/计数审计，不训练结果模型。

### T1 高维内部实现

只修改完成本协议所需部分：高维 PhiSN、长度尺度子采样、`KNNFeatureRegion`、统一
route mask 和快照。公共 API 不新增 detector/validator/support-builder 插件接口。

### T2 测试脊柱

新增至少以下测试：

- 24 维 `fit/observe/evaluate/promote/forward` smoke；
- KNN region 的 validation 覆盖、ID guard 排斥和 unknown 排斥；
- 高维区域外 exact fallback；
- 晋升后 region/prototype/GP snapshot 不随 observe 改变；
- 五种方法共享同一 checkpoint、RFF map 和数据顺序。

### T3 前置硬门

运行 P0–P3。任何前置失败立即按 §9 结束，不运行 50-run 正式矩阵。

### T4 正式矩阵与判决

运行 50 runs，一次生成原始 pickle/JSON、逐 seed 表和机械 verdict；判决脚本不得读取
手工编辑后的汇总表。

### T5 展示

只制作三张主图：

1. 32D embedding 的固定 PCA 二维投影 + ID/new/unknown + 授权 mask；
2. 稳定/反转流的阶段性 RMSE 与晋升事件；
3. 五方法的 `new-domain gain` 对 `ID/unknown interference` 安全—塑性图。

最终交付：

```text
docs/Phase3_Plan.md
results/PHASE3_PRECONDITIONS.md
results/PHASE3_RESULT.md
results/phase3_data_manifest.json
results/phase3_matrix.json
results/phase3_metrics_table.md
results/phase3_embedding.png
results/phase3_stream_behavior.png
results/phase3_safety_plasticity.png
examples/real_stream_cmapss.py
```

## 11. 允许与禁止的实现判断

允许：修复明确 bug、减少等价计算、缓存 checkpoint、把 CPU 张量位置处理正确、为同一
公式增加数值稳定性。所有会影响判决的修复必须保留首轮输出，并在重跑前写 amendment。

禁止：为追求一次 PASS 引入 LSTM/Transformer、更换数据集、增加隐藏特征、按结果挑
工况、修改能力阈值或把 unknown 数据加入区域构造。若 GP 容量失败，它就是
`PIVOT_LEARNER` 的真实结论，后续容量研究另开 Phase 4。
