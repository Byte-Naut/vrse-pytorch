# Phase 3B 手动运行与验证

> 代码状态：Phase-3B 已执行并冻结；本文仅用于独立复现。  
> 主协议：`docs/Phase3_Plan.md`  
> 修订：`docs/Phase3B_Amendment.md`

Phase-3A 的无效产物已保存在 `results/run1_invalid_normalization/`。不要删除或覆盖该
目录，也不要把其中的 `PIVOT_CORE` 与 Phase-3B 合并。正式冻结产物及校验哈希见
`results/PHASE3B_SNAPSHOT.md` 和 `results/phase3b_snapshot.sha256`；任何复跑应输出到
新目录并与冻结产物比较，不得就地覆盖。

## 1. 冻结数据与来源

优先直接使用已有的原始解压目录，避免把本地重打包 ZIP 的哈希误写成官方哈希：

```powershell
python -m experiments.phase3_prepare_data `
  --source data/CMAPSSData `
  --source-origin user-attested-original-extraction
```

检查 `results/phase3_data_manifest.json`：

- `protocol_revision == "phase3b-discovery-global-normalization-v1"`；
- `source_kind == "extracted_directory"` 且列出逐文件 SHA-256；
- `normalization.passed == true`；
- `normalization.floor_features_with_outside_variation == []`；
- `units == 260`，三个目标工况互异。

`official_archive_sha256: null` 是刻意的：当前没有权威官方压缩包哈希可供声明。

## 2. 运行前置硬门

```powershell
python -m experiments.phase3_preconditions
```

该命令会运行既有测试与两个 Stage-4C 回归锚点，再训练并冻结五个共享 checkpoint。
只有 `results/PHASE3_PRECONDITIONS.md` 显示 `READY_FOR_MATRIX` 时才能继续。正式运行
不要使用 `--skip-phase2-gates`。

额外核对：每个 checkpoint 和 `phase3_preconditions.json` 均携带 Phase-3B 协议标识，
且 normalization source 指向无标签 discovery units 1–20。

## 3. 运行正式矩阵

```powershell
python -m experiments.phase3_matrix
python -m experiments.phase3_verdict
python -m experiments.phase3_visualize
```

脚本会拒绝 Phase-3A 的旧 precondition、checkpoint 或 matrix。判决仍只读取原始
pickle，不读取手工编辑的 Markdown 表。

## 4. 示例与交付物

```powershell
python -m examples.real_stream_cmapss
```

主要产物不变：manifest、preconditions、五个 checkpoint、matrix pickle/JSON、verdict、
结果表与三张图。若再次发现协议缺陷，先保存整轮产物并新增 amendment，不得事后改写
门槛或挑选 seed。
