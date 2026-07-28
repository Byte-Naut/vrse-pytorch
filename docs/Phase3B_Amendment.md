# Phase 3B 协议修订：跨工况无标签归一化

> 冻结日期：2026-07-29  
> 状态：**EXECUTED / FROZEN / PASS**  
> 协议标识：`phase3b-discovery-global-normalization-v1`

## 1. Phase-3A 的有效判定

Phase-3A 的 50-run 矩阵按原协议机械得到 `PIVOT_CORE`，但该判决不具有科学解释力，
应记为 `INVALID_PROTOCOL_NORMALIZATION`，而不是 VRSE 失败。

原协议只用单一 ID 工况的 `ID-fit` 行估计 24 维尺度，并把标准差下限设为 `1e-6`。
实际 FD002 中六个通道在该工况内近似或完全不变、在其他工况却发生真实位移，导致
新工况的绝对 z-score 最高约 `1.76e8`，冻结 baseline 的输出约为 `1.47e6`，而目标被
限制在 `[0, 125]`。因此：

- H1/H3 的精确隔离可能只是对灾难性数值输入的机械回退；
- H2 的反转流错误晋升无法区分 VRSE 决策失败与归一化伪影；
- H4 衡量的是“较少灾难”而非可用效益；
- P2 的 100% 拒绝不能被解释为已校准的认知不确定性。

原始产物保存在 `results/run1_invalid_normalization/`。不得将其中的 `PIVOT_CORE` 用作
支持或反驳 VRSE 的结果。

## 2. 唯一协议改动

冻结 units 1–20 继续承担 `regime_discovery`，但在工况发现之外，再用其**全部无标签
24 维输入**计算一套公共均值与标准差：

```text
mu_j    = mean(x_j | unit in 1..20)
sigma_j = max(std(x_j | unit in 1..20), 1e-6)
z_j     = (x_j - mu_j) / sigma_j
```

这 20 个 unit 已在 Phase-3A 中预先隔离，不进入 ID fit/calibration/guard、shadow
observe、promotion validation 或 post-decision test；尺度估计不读取 RUL 标签。所有
五个 seed 和全部方法使用完全相同的 `mu`、`sigma`。

这是对测量坐标系的修复，不是模型调参：它利用冻结 discovery 样本覆盖六个工况，
从而保留工况间位移；不会像 per-regime normalization 那样抹掉 OOD 信号，也不会像
删除 ID 零方差通道那样删除可能有意义的运行条件。

## 3. 新增 P0 硬门

在训练 baseline/PhiSN 之前必须满足：

1. 归一化统计量的来源严格等于 units 1–20，且五个 seed 逐位一致；
2. 所有归一化输入有限；
3. 任一 discovery 标准差低于 `1e-6` 的特征，在完整原始 FD002 输入中的跨度也必须
   小于 `1e-6`，否则 `INVALID`；
4. manifest 报告全部原始行上的最大绝对 z-score，仅作审计，不另设事后性能阈值；
5. checkpoint、precondition 和 matrix 均携带上述协议标识；旧版产物必须 fail closed。

第 3 条直接排除 Phase-3A 的失败机制，不依据模型得分或晋升结果选择阈值。

## 4. 数据来源修订

首轮运行使用的是从已有原始文件本地重打包的 ZIP，故该 ZIP 的 SHA-256 不是 NASA
官方压缩包标识。Phase-3B 默认直接读取 `data/CMAPSSData/`，逐文件记录 SHA-256，
并把来源声明与哈希分开：`source_origin` 是操作者声明，`source_files_sha256` 才是可复核
身份。除非另有权威记录，`official_archive_sha256` 保持 `null`。

## 5. 保持冻结的内容

数据集、工况发现、ID/new/unknown 选择、五个 seeds、unit 角色、RUL cap、baseline、
PhiSN/SNGP、GP 容量、五种方法、KNN 区域、两条流、50-run 预算及 P1–P3/H1–H4
门槛全部不变。此前两个实现修复继续保留：谱归一化训练后的无梯度估计器收敛，以及
保存 checkpoint 前将已冻结的谱参数化烘焙为普通权重。

## 6. 判决解释

Phase-3B 是一次新的预注册复跑。结果可以判定原始 Phase 3 研究问题；它不能与
Phase-3A 数字合并、择优或声称为同一次首次运行。若新的 P0 失败则仍为协议无效；
若 P0 通过，之后严格沿用 `docs/Phase3_Plan.md` 的既有机械判决链。
