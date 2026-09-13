# S4 结果 — E5 多视野统一输出 / E6 跨域零样本迁移（EnergyTS 对标）

> 环境：A100-PCIE-40GB，单卡。
> 口径：E5 WikiText-103、2000 步、3 seeds、horizons {8,32,128}；E6 合成三域 leave-one-domain-out、2000 步、2 seeds。
> 原始数据：服务器 `/root/C1/results_s4/*.jsonl`。

## 结论速览

| 实验 | 结果 |
|---|---|
| E5 多视野统一 vs 专用 | 🟢 **首个偏正面信号**:共享主干在长视野(32/128)更好,短视野(8)专用略好 |
| E6 跨域零样本 | 🟡 混合:events 迁移成立,ar 反向,periodic 混合 |

---

## E5 — 多视野统一输出（unified vs dedicated，2000 步 × 3 seeds）

`unified` = 单主干 + 每视野一个头(联合训练)；`dedicated` = 每视野单独一个模型(同单模型预算)对照。指标:各视野 val PPL(越低越好)。

| horizon | unified (mean±std) | dedicated (mean±std) | 胜者 |
|---|---|---|---|
| h=8 | 1258.4 ± 15.2 | **1226.9 ± 1.2** | dedicated |
| h=32 | **1348.0 ± 11.0** | 1497.3 ± 182.1（seed0 离群 1706） | **unified** |
| h=128 | **1412.9 ± 7.8** | 1474.6 ± 9.1 | **unified** |

- **共享主干在长视野(32/128)明显更好**(h=128 稳定领先约 4%);短视野 h=8 专用略胜约 2.5%。
- h=32 的 dedicated 均值被 seed0 离群值(1706)拉高;即便剔除,unified 仍约领先 3%。
- **部分支持 EnergyTS"长短时路径统一架构"**:统一主干不仅不牺牲长视野,反而在长视野更优。

## E6 — 跨域零样本迁移（leave-one-domain-out，2000 步 × 2 seeds）

三域:periodic / ar / events。指标:留出域的 zero-shot CE(越低越好),对照 random-init 与 single-domain。

| held-out | zero_shot_ce | random_init_ce | gain_vs_random | gain_multi_vs_single |
|---|---|---|---|---|
| periodic | 5.42 / 6.78 | 5.55 / 5.51 | +0.12 / **−1.27** | +0.02 / −0.13 |
| ar | 5.62 / 5.79 | 5.51 / 5.48 | −0.11 / −0.31 | +0.53 / −0.21 |
| events | 4.11 / 4.13 | 5.48 / 5.52 | **+1.37 / +1.39** | +0.24 / **+2.27** |

- **events 域迁移稳定成立**:多域预训练在留出 events 上比随机初始化好 ~1.37 CE,且多域显著优于单域。
- **ar 域迁移失败**(negative):多域预训练反而比随机差。
- **periodic 混合**,且 seed 间方差大(seed1 严重负)。
- 结论:**跨域零样本迁移不是普遍能力,而是域依赖的** —— 稀疏事件流(events)可迁移,平滑自回归(ar)不可。

---

*S4 = 首个偏正面结果(E5 统一主干利于长视野)+ 域依赖的零样本迁移(E6)。下一步:S5 扩大 seed 确认 E5/E6。*
