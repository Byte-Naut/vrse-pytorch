# Phase 3A 首轮结果

> 科学判定：**INVALID_PROTOCOL_NORMALIZATION**  
> 原机械输出：`PIVOT_CORE`（不得作机制证据）

Phase-3A 按冻结协议完成了前置门与 50-run 矩阵，但 ID-fit-only 标准化把六个仅在
ID 工况内零方差的特征除以 `1e-6`。这些特征在新工况发生真实位移，产生最高约
`1.76e8` 的 z-score，并使 baseline 原始输出约为 `1.47e6`，远离 `[0,125]` 的目标域。

因此全部 H1–H4 数字均受同一数值伪影支配，无法回答 VRSE 在高维真实任务上是否
成立。它既不是 PASS，也不是 VRSE 的 `PIVOT_CORE`；正确状态是协议失效、机制未判定。

原矩阵、图和原机械结果完整保存在 `results/run1_invalid_normalization/`。Phase-3B
只修订无标签尺度估计与来源追踪，详见 `docs/Phase3B_Amendment.md`。
