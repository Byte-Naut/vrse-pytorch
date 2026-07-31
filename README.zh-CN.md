# VRSE

### 让 PyTorch 模型持续学习，同时不破坏已验证的能力。

VRSE（Validated Regional Support Expansion）是一个持续学习（Continual
Learning）内核研究原型：既有模型保持为默认服务，候选行为在隔离环境中训练，只有通过独立留出数据评估后，才会在证据支持的输入区域内获得服务权限。

> **状态：research alpha / reference implementation。** 当前版本验证了标量回归的生命周期与路由机制，并完成了一项 24 维代表性实验。

[English README](README.md) · [运行 Quickstart](#quickstart) ·
[体验完整生命周期](#体验完整生命周期) · [查看证据](#当前实验证据) ·
[理论边界](docs/THEORY.md)

## 核心功能：在新的数据区域更新模型，在旧的数据区域回退基线 · 自主学习 · 自动判断

![VRSE notebook 结果：promotion 仅改变授权区域](docs/assets/vrse-regional-adaptation-demo.png)

*notebook 演示：在一维拟合任务中，面对持续到达的数据，VRSE 改善了新工况，而被保护的旧工况和相邻未知输入保持 baseline 不变。*

## 快速开始

```bash
python -m pip install -e .
python -m examples.quickstart
```

在持续学习系统中应用：

```python
from vrse import VRSEConfig, VRSEModel

# 1. 部署 — 冻结已知的安全服务
model = VRSEModel.wrap(baseline, VRSEConfig(preset="regional_regression_highdim"))
model.fit(x_id, y_id, x_id_calibration)  # in-domain 数据（已知、可信的旧数据）

# 2. 在线服务 — 自动路由；不熟悉的输入暂存缓冲池
for x_live in stream:
    y_hat = model(x_live)                     # 授权区域走专家网络，其余退回 baseline
    unfamiliar = model.review_mask(x_live)     # 当前支持范围外 → 留待审查
    buffer.store(x_live[unfamiliar])

# 3. 审查 — 标签到达后（可能数小时或数天后）
buffer.attach_labels(sample_ids, y_delayed)
if buffer.ready():
    learn, exam = buffer.take_labeled_disjoint()
    model.observe(learn.x, learn.y)           # 候选模型隔离训练
    proposal = model.evaluate(                # 新数据考试 + 旧数据护栏
        exam.x, exam.y, guard_x=x_id_guard,
    )
    model.promote(proposal)                   # 通过 → 授权；失败 → 丢弃
```

观测与标签是两个独立事件，通过 sample ID 关联。外层系统管理缓冲池和调度；VRSE
负责学习、审查、授权和路由。

确定性示例直接输出用户关心的事件：

```text
Candidate learned in isolation       yes
Served model changed before review   no
Useful candidate promoted            yes
Harmful candidate promoted           no
Old behavior changed                 no
Unknown inputs changed               no
Revoke restored previous snapshot    yes
```

## 核心理论机制：验证区支持扩展（Validated Regional Support Expansion, VRSE）

**工业痛点与解法。** 在安全关键或稳定的生产环境中，直接对全局模型 $f_0(x)$ 进行在线更新会将两个决策强制绑定：学习新知识，以及立刻改变所有输入的预测行为。VRSE 在形式化上将**“学习（Learning）”**与**“服务授权（Permission to Serve）”**彻底解耦。

**服务函数的形式化。** 设 $f_0: \mathcal{X} \rightarrow \mathbb{R}$ 为不可变的既有 baseline。获得标签的审查数据仅流入一个被严格隔离的影子候选模型 $e^\star(x)$ 中。经过留出数据评估后，及格的候选模型会被晋升为第 $t$ 代部署快照，包含一个冻结的专家 $e_t$ 和一个明确的授权区域 $A_t \subseteq \mathcal{X}$。

此时的系统服务函数在数学上被严格保证为：

$$
F_t(x) =
\begin{cases}
f_0(x) + e_t(x), & x \in A_t, \\[4pt]
f_0(x), & x \notin A_t.
\end{cases}
$$

**异步自动化的生命周期。**

1. **数据捕获（审查掩码）：** 基线模型维持一个不确定性分数 $u_0(x)$ 以及校准阈值 $\tau_0$。当实时输入既超出了基线的熟悉范围，又不在当前路由覆盖区域内时，前台系统拦截该输入，将其存入缓冲池等待延迟标签：

$$
R_t(x) = \mathbf{1}[u_0(x) > \tau_0]\,\mathbf{1}[x \notin A_t]
$$

2. **晋升管线：** 标签到达后，由外部 MLOps 系统驱动状态机流转：

$$
D_{\mathrm{obs}} \xrightarrow{\mathrm{observe}} e^\star \xrightarrow[D_{\mathrm{val}}, D_{\mathrm{guard}}]{\mathrm{evaluate}} P^\star \xrightarrow{\mathrm{promote}\;\text{if}\;P^\star.\mathrm{passed}} (e_t, A_t)
$$

`observe()` 仅更新影子专家，不触碰在线服务。`evaluate()` 利用留出的新工况数据（$D_{\mathrm{val}}$）考核候选专家的效用，并检查建议区域是否会让给定的旧数据护栏（$D_{\mathrm{guard}}$）进入专家路由、或改变这些输入的输出。`promote()` 仅将通过考核的快照原子化上线。

**五项系统级不变量。** 该代码实现从结构上保证了：(i) **影子非干扰**（后台学习永远不改变前台输出）；(ii) **精确回退**（在 $A_t$ 之外，附加残差在代数上绝对为零）；(iii) **提案绑定**（过期或伪造的考试记录会被拒绝）；(iv) **原子化晋升**；(v) **单步回滚**。

关于完整的数据角色约束、数学定义及统计学声明边界，请参阅 [`docs/THEORY.md`](docs/THEORY.md)。

## 体验完整生命周期

![VRSE 生命周期](docs/assets/vrse-lifecycle.svg)

1. baseline 持续服务，新工况的有标签数据开始到达；
2. shadow candidate 在隔离状态下学习，当前服务输出不变；
3. 独立验证集考察新能力，guard 集检查对旧工况的侵入；
4. 有用候选连同授权区域一起冻结并上线；
5. 旧工况和相邻未知工况继续精确使用 baseline；
6. 有害候选被拒绝，`revoke()` 可恢复上一快照。

> 学习可以持续进行；新行为是否进入服务，仍由明确且基于证据的门控决定。

可执行的图形化说明见
[`notebooks/vrse_lifecycle.ipynb`](notebooks/vrse_lifecycle.ipynb)。

## 可能的应用方向

当前完成验证的任务包括 C-MAPSS FD002 工业仿真实验。跨领域和不同模型的计划见
[`docs/BENCHMARK_PLAN.md`](docs/BENCHMARK_PLAN.md)。

未来有潜力的应用方向包括：工业状态监测、传感器回归、网络调度等需要安全、可解释的持续学习能力的领域。

## 实验证据

代表性实验使用 NASA C-MAPSS FD002 涡扇发动机剩余寿命仿真 benchmark（每个输入 3
项运行设置 + 21 项传感器值）。完整发动机被分配到互不重叠的 fit、calibration、
observation、validation、guard 和 post-decision 角色。

| 问题 | 结果 |
|---|---:|
| Utility：稳定新工况 RMSE，baseline → VRSE | 96.18 → **21.61** |
| Utility：平均 RMSE 降幅 | **77.5%** |
| Promotion：有用候选通过 | **5/5** |
| Rejection：反转候选错误通过 | **0/5** |
| Coverage：新工况 expert 路由覆盖 | **93.0–96.0%** |
| Non-interference：ID / 相邻未知 expert 路由 | **0.0% / 0.0%** |

![效用与拒绝](results/cmapss_fd002_stream_behavior.png)

稳定工况下 VRSE 恢复了全局 shadow expert 的大部分效用；标签反转时，无门控方法上线了有害更新，而 VRSE 拒绝了候选。

![授权覆盖与非干扰](results/cmapss_fd002_embedding.png)

授权区域覆盖了大部分新工况，同时旧工况和相邻未知工况保持回退——排除了"拒绝所有输入"的虚假非干扰。

![安全—可塑性折中](results/cmapss_fd002_safety_plasticity.png)

全局适应有用但干扰全局，静态拒绝无干扰但不适应；VRSE 在本实验中位于有用且低干扰的角落。

相关证据：[`结论快照`](results/CMAPSS_FD002_SNAPSHOT.md) ·
[`决策机制`](results/CMAPSS_FD002_RESULT.md) ·
[`逐 seed 指标`](results/cmapss_fd002_metrics.md) ·
[`原始矩阵`](results/cmapss_fd002_matrix.json) ·
[`实验协议`](docs/CMAPSS_FD002_PROTOCOL.md)

## 如何参与

- **Reproduce**：在不同系统或硬件上复现 quickstart 和冻结实验；
- **Challenge**：提交能够击穿当前区域授权规则的最小反例；
- **Extend**：实现分类、真实在线仿真、多轮晋升、多 expert 或新模型 adapter；
- **Compare**：在同一数据流中比较全局微调、replay、regularization、reject-all
  和标准 CL 工具。

具体工作包见 [`ROADMAP.md`](ROADMAP.md)，贡献方式见
[`CONTRIBUTING.md`](CONTRIBUTING.md)，复现入口见
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md)。

本项目使用 [Apache License 2.0](LICENSE)。项目名称不表示安全认证，也不表示 NASA
或其他机构背书。
