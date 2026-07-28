# VRSE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21653095.svg)](https://doi.org/10.5281/zenodo.21653095)

### 让在线模型持续学习，但不让未经验证的更新直接上线

**验证式区域支持扩张（Validated Regional Support Expansion，VRSE）** 是一个
PyTorch 科研原型库，用于实现一种保守的在线适应方式：新数据先在隔离的影子专家中
学习；候选必须通过独立考试；通过后也只获得一块有证据支持的输入区域的服务权限。
其余输入始终由冻结基线原样处理。

[English README](README.md) · [快速体验](#快速体验) ·
[真实任务结果](#24-维真实任务验证) · [完整复现](#复现冻结结果)

![VRSE 生命周期](docs/assets/vrse-lifecycle.svg)

## 它解决什么问题

普通在线学习把两件事绑在一起：模型一边从新数据学习，一边立即改变用户正在使用的
服务。一旦新数据来自短暂变化、错误标签或尚未理解的区域，更新可能影响整个模型。

VRSE 把“学习”和“获得服务权限”拆开：

1. 复制并冻结既有基线，使其始终作为默认服务；
2. 新标签只更新影子残差专家，不改变线上输出；
3. 使用独立的新区域验证集考察能力，并用旧区域 guard 检查是否侵犯既有行为；
4. 通过考试的候选被连同授权区域一起冻结为部署快照；
5. 区域外输入继续获得与原基线逐点相同的输出。

它并不要求后台不能学习，而是要求任何学习结果必须先证明自己，再获得与证据范围
相匹配的有限权限。

## 快速体验

当前版本是从源码安装的 research alpha：

```bash
python -m pip install -e .
python -m examples.quickstart
```

核心接口只有一个清晰的生命周期：

```python
from vrse import VRSEConfig, VRSEModel

model = VRSEModel.wrap(
    baseline=baseline,
    config=VRSEConfig(preset="regional_regression_highdim"),
)

model.fit(x_id, y_id, x_id_calibration)       # 冻结既有服务
model.observe(x_new, y_new)                   # 只训练影子候选
proposal = model.evaluate(x_validation, y_validation, guard_x=x_id_guard)
promoted = model.promote(proposal)            # 原子晋升或拒绝
y_hat = model(x)                              # 区域专家或精确回退
```

`fit`、`observe`、`evaluate` 和 `guard_x` 对应不同的数据角色。若重复使用同一批数据，
晋升考试就失去了独立证据的意义。完整的确定性示例见
[examples/quickstart.py](examples/quickstart.py)。

该示例的已验证输出为：

```text
VRSE quickstart
  isolated learning max output change : 0.000e+00
  promotion passed                    : True
  new-region route fraction           : 1.000
  new-region RMSE before -> after      : 2.500 -> 0.003
  old-region max fallback difference  : 0.000e+00
  unknown max fallback difference     : 0.000e+00
```

## 实现层面提供什么

- `observe()` 不会改变当前正在服务的部署快照；
- 晋升建议会绑定被考试的基线、配置、候选和授权区域，过期或重复建议会被拒绝；
- 晋升将 GP 后验与区域一起原子冻结；
- 授权区域外，输出精确等于冻结基线，而不只是平均意义上“基本不变”；
- 支持一步撤销，恢复上一份部署快照。

这些是软件与路由性质，不代表基线本身一定安全，也不代表统计上能够识别所有未知
输入或保证专家在证据范围之外仍然可靠。

## 24 维真实任务验证

代表性实验采用 **NASA C-MAPSS FD002**：一个用于涡扇发动机剩余寿命预测的工业仿真
benchmark。每个输入包含 3 项运行设置与 21 项传感器值。它是仿真数据，不是实际
飞机现场数据。

实验严格按发动机划分数据角色，运行五个固定 seed，并配对两条输入完全相同的流：

- **稳定工况**：新工况规律持续成立，应当学会并晋升；
- **反转工况**：验证阶段标签被机械反转，应当拒绝晋升。

| 五个 seed 的冻结结果 | 冻结基线 | VRSE |
|---|---:|---:|
| 稳定新工况平均 RMSE | 96.18 | **21.61** |
| RMSE 降幅 | — | **77.5%** |
| 稳定候选晋升 | — | **5/5** |
| 反转候选错误晋升 | — | **0/5** |
| 新工况专家路由覆盖 | — | **93.0–96.0%** |
| ID / 相邻未知专家路由 | — | **0.0% / 0.0%** |

![稳定适应与反转负对照](results/phase3_stream_behavior.png)

稳定流的收益不是靠“全部拒绝”得到的：绝大多数受支持的新工况样本确实进入了晋升
专家，误差降低约 4.5 倍。反转流中，无门在线方法上线了有害更新，而 VRSE 在五个
seed 上全部拒绝并精确回退。标签反转同时使候选变差、也使基线分母变得更容易，因而
它应被理解为有效的受控拒绝测试，而不是“只有候选退化导致拒绝”的证明。

<p align="center">
  <img src="results/phase3_embedding.png" width="49%" alt="冻结特征空间与区域授权">
  <img src="results/phase3_safety_plasticity.png" width="49%" alt="五种方法的安全与适应折中">
</p>

左图显示晋升区域覆盖新工况，同时 ID 与相邻未知簇保持回退；右图展示了三类选择：
全局适应有收益但会全域干扰，静态拒绝没有干扰但无法学习，VRSE 在本实验中同时取得
较高新工况收益与零保护区路由。

完整证据见 [冻结快照](results/PHASE3B_SNAPSHOT.md)、
[机械判决](results/PHASE3_RESULT.md)、[逐 seed 指标](results/phase3_metrics_table.md)
和 [冻结协议](docs/Phase3_Plan.md)。

## 复现冻结结果

仓库不重新分发 C-MAPSS。请从
[NASA 数据页](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
取得数据并解压至 `data/CMAPSSData/`，然后执行：

```bash
python -m pip install -e ".[benchmark,test]"
python -m experiments.phase3_prepare_data \
  --source data/CMAPSSData \
  --source-origin user-attested-original-extraction
python -m experiments.phase3_preconditions
python -m experiments.phase3_matrix
python -m experiments.phase3_verdict
python -m experiments.phase3_visualize
```

只有前置判决为 `READY_FOR_MATRIX` 才能继续。Windows 命令、来源说明和核验步骤见
[Phase3_Runbook.md](docs/Phase3_Runbook.md)。冻结产物由
[phase3b_snapshot.sha256](results/phase3b_snapshot.sha256) 唯一标识。公开版本包含清单所列的
冻结矩阵和 checkpoint；由 C-MAPSS 生成的预处理数组不随仓库分发，需从源数据重新生成。

## 当前边界

`0.1.0a1` 只支持有监督标量回归、单个活动区域残差专家、CPU、SNGP 风格距离感知
特征，以及一维观测跨度或高维 KNN 特征区域。当前结果不能推广为：

- 已普遍解决持续学习或灾难性遗忘；
- SNGP 不确定性能够发现所有分布变化；
- 已获得控制、临床、金融等高风险系统的安全认证；
- 已验证分类、延迟标签、对抗流、多专家并存或多轮区域组合；
- 已取得 C-MAPSS 精度 SOTA。

项目保留了中途的负面实验和无效运行，因为这些记录构成最终设计可信度的一部分。
简洁研究定位与证据边界见 [RESEARCH_SCOPE.md](docs/RESEARCH_SCOPE.md)；更长的理论与
失败路径文档继续保留在 `docs/` 中作为研究过程档案，不作为首次使用的前置阅读。

## 发布状态

核心生命周期与 Phase‑3B 已冻结。`0.1.0a1` 是与该结果对应的首个公开科研软件版本，
属于 research preview：不承诺 API 稳定，也不代表生产就绪。该版本的不可变归档见
[Zenodo](https://doi.org/10.5281/zenodo.21653095)。

发布者为 **Byte-Naut**，一个独立的开源研究身份。

> Byte-Naut. (2026). *VRSE-PyTorch v0.1.0a1: Validated Regional Support Expansion
> for Safe Online Adaptation* [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.21653095

VRSE 采用 [Apache License 2.0](LICENSE) 开源。项目名和包名仅用于标识这一科研实现，
不表示获得 NASA 或其他机构的安全认证或背书。
