# S1 结果 — E1 液态 GRPO / E2 MTP / E3 时间尺度扩容

> 环境：A100-PCIE-40GB，单卡，约 6 小时。
> 口径：E1 parity 30000 步 × 3 seeds；E2 WikiText-103 2000 步 × 3 seeds（125M 液态主干）；E3 parity 3000 步 × 3 seeds。
> 原始数据：服务器 `/root/C1/results_24h/*.jsonl`（未入库，`*.jsonl` 在 .gitignore 内）。

## 结论速览

| 实验 | 假设 | 结果 |
|---|---|---|
| E1 液态 GRPO | 纯 RL 从零涌现长度外推 | ❌ 否定（3/3 seed ≈ 随机） |
| E2 MTP 前瞻 | token/状态前瞻提升 PPL | ❌ 否定（mtp ≈ baseline；state-mtp 更差） |
| E3 时间尺度扩容 | 扩容 + top-k 是"免费容量维度" | ⚠️ 不确定（地板效应，任务未学会） |

---

## E1 — 液态 GRPO（parity 长度外推，3 seeds × 30000 步）

评测：整序列准确率，L ∈ {32,48,64,96,128}，训练 L ∈ {1..32}。

| seed | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|
| 0 | 0.484 | 0.516 | 0.523 | 0.500 | 0.438 |
| 1 | 0.516 | 0.484 | 0.531 | 0.500 | 0.398 |
| 2 | 0.625 | 0.539 | 0.461 | 0.531 | 0.430 |

全部 ≈ 0.5（二分类随机基线）；`mean_r` 全程在 0.41–0.73 震荡、无趋势。**纯 GRPO 从随机初始化没有涌现 parity 机制。** 与监督臂"seed0 达标、seed1/2 随机"的模式一致 —— 支持"液态 RL 需要先有机制（热启）"的判断。下一步：`grpo_finetune`（从监督 checkpoint 热启）。

## E2 — MTP 前瞻（WikiText-103，2000 步，3 seeds）

val PPL（gpt2 vocab 50257，125M 液态主干）：

| 模式 | seed 0 | seed 1 | seed 2 | 均值 |
|---|---|---|---|---|
| baseline（单步 CE） | 151.57 | 152.67 | 154.55 | **152.93 ± 1.49** |
| mtp（k=3） | 151.64 | 153.32 | 154.31 | **153.09 ± 1.35** |
| state-mtp（+液态状态前瞻） | 160.89 | 155.33 | 158.85 | **158.35 ± 2.79** |

mtp 与 baseline 打平（无增益）；state-mtp 反而差约 3.5%。**MTP 前瞻在此规模/预算下未带来 PPL 收益。**

> 实现修复（本轮）：原 `MTPWrapper` 的 `head[j]` 预测**同位置** token（`h_t → x_t`，信息泄漏），baseline 训练 loss 坍缩到 ~0，且评估用未训练的 base `lm_head` → val PPL ~6e4 无意义。修复为 `head[j]` 预测 `x_{t+j+1}`、评估用 `token_heads[0]`。修复后 baseline step500 loss 0.29 → 5.89，val PPL 63k → 152。见 commit `fix(e2)`。

## E3 — 时间尺度扩容（parity，3000 步，5 配置 × 3 seeds）

配置：`n_scales ∈ {5,16}` × `topk ∈ {dense,2,5,8}`。**所有配置在同一 seed 下 accs 完全相同**，全部 ≈ 0.5。

探针验证（`d_model 208/4L/13H`）：`n_scales 5→16` 确实改变模型（参数 1.11M → 1.27M，输出 maxdiff 0.45），topk 亦生效。故相同 accs **不是旋钮失效，而是任务未学会（地板效应）**——3000 步 + 未开 `selective_decay` 的配置无法学会 parity，实验无法区分配置。

> 实现修复（本轮）：`--topk` 原先静默失效（应写 `sparse_resonance_kernel` + `sparse_resonance_top_k`，脚本误设 `m.topk`）；`d_model` 默认 256 不被 13 整除。见 commit `fix(e3)`。
> 下一步：套用 M1 已验证 parity 配方（`d_model 104 / 2L / 4H / 2KV`、`selective_decay=True`、30000 步）先让任务学会，再比 scales。

---

*S1 = 三个负/不确定结果。纯液态激进路线在「RL 涌现推理」「MTP 前瞻」「时间维度扩容」上均暂无证据支持 —— 与 C1/M1 一贯的诚实证据纪律一致。*
