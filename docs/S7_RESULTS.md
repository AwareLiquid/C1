# S7 结果 — E5 视野扩展到 512 + 头权重模式消融

> 环境：A100-PCIE-40GB，单卡。原始数据：`/root/C1/results_s7/*.jsonl`。

## 结论速览

| 项 | 结果 |
|---|---|
| E5 视野 = 512 | ⚠️ **无效**:视野必须 < seq_len;512 时该头无目标 |
| E5 dedicated | ❌ **崩溃**(已修复,见 `fix(e5)`) |
| 头权重模式消融 | 🟢 **inv_sqrt(默认)最优** |

---

## E5 unified（horizons 8/32/128/512，4000 步，3 seeds）

| horizon | unified PPL (mean, n=3) |
|---|---|
| h=8 | 1130.5 |
| h=32 | 1231.0 |
| h=128 | 1313.6 |
| h=512 | **1.000（退化,无效）** |

- **h=512 无效**:`seq_len=512` 时 `T - h = 0`,该头没有目标,返回退化值 PPL=1.0。
- 视野必须严格 < seq_len。要测 512 需 `seq_len ≥ 1024`。
- h=8/32/128 有效,且与 S5/S6 趋势一致。

## E5 dedicated — 崩溃（已修复）

dedicated(单视野模型)在 h=512 时 `loss` 变成 Python float(`total=0.0` 未被任何有效头累加)→ `loss.backward()` 抛 `AttributeError`。已在 `fix(e5)` 修复:
1. `total` 初值改为张量(`hidden.sum() * 0.0`),保证 backward 可用;
2. `train_eval` 过滤 `h < seq_len` 的视野。

## 头权重模式消融（horizons 8/32/128，2000 步，seed 0）

| weight_mode | h=8 | h=32 | h=128 |
|---|---|---|---|
| **inv_sqrt（默认）** | **1249.3** | **1343.2** | **1411.9** |
| inv | 1305.9 | 1398.5 | 1457.4 |
| equal | 1369.9 | 1419.3 | 1462.0 |

- **inv_sqrt 最优**(各视野都最低),`inv` 次之,`equal` 最差。
- 说明"近视野权重更高、且随视野以 1/√h 衰减"的加权是合理设计;完全均权会牺牲所有视野。

---

*S7 = 发现并修复 E5 的视野边界 bug(h ≥ seq_len);确证默认头权重 inv_sqrt 最优。dedicated 需在修复后重跑(S7b)。*
