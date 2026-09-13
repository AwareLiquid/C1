# C1 — AwareLiquid Cortex

**第二代类脑认知架构：从"液态神经网络"到"液态认知栈"**

Cortex（C1）不是 Transformer 的补丁，也不是 MT-LNN 的简单升级——它是从大脑皮层分层结构出发的**完整认知架构**，把"感知流处理、O(1) 工作记忆、稀疏时间路由、预测编码学习、强化学习涌现"组织成一条五层认知栈。

> **与 M3 的关系（2026-08-18 定性）**：C1 = M3 的**公开愿景版（纯液态激进变体）**，
> 回答"不要注意力、液态栈单独能走多远"（长上下文 + 端侧）；M3（私有）= 证据驱动
> 混合版（窗口注意力 + 液态 + fast-weight × 三轴），回答"如何逼近 Gemini 3 问答
> 质量"。同一架构两种表达，证据冲突时以 M3 实测为准。

> 核心信念：智能不是"下一个 token 预测得更准"，而是**像大脑一样——只对当下真正重要的信号放电，用 O(1) 状态记住本质，用预测误差驱动学习，用奖赏信号涌现推理。**

---

## 为什么是 C1（Cortex）？

MT-LNN 证明了"微管液态动力学"能在小规模上跑通：O(1) 内存（8063× @1M 上下文）、跨窗口记忆（0.56 vs 注意力 0.000）、长文本抗噪（×2.0 @T=229）。但它仍然是"用液态网络替换注意力层"——**架构还停在"层"的层面**。

C1 的跃迁：**从"替换层"到"重做认知栈"**。

大脑皮层分六层，每层分工不同（输入、联合、工作区、预测、输出、自上而下调节）。C1 用五层认知栈对应这个结构——**不是把注意力换成液体，而是让整个信息流按皮层的组织方式运转**。

```
皮层分层          ↕          C1 五层栈
L4  输入层     →    L1 感知流原生处理（SensoryFrontend）
L2/3 工作区    →    L2 O(1) 液态工作记忆
L5  联合层     →    L3 稀疏时间尺度路由（13原丝 × N尺度）
L2/3 预测      →    L4 多尺度预测编码
L5/6 输出+调节  →    L5 RL 推理涌现 + 系统1/2双轨
```

---

## 与 DeepSeek / Kimi 的对照（借鉴但不跟随）

DeepSeek 和 Kimi 在同一堵墙前给出了两个答案：DeepSeek 优化"单 token 成本"（MLA 省缓存、MoE 省参数、FP8 省训练），Kimi 优化"整段上下文利用率"（长链推理、原生多模态）。**两者共同确认了：稀疏激活、预测式学习、RL 涌现是行业共识方向。**

C1 的翻译：

| 行业方向（DeepSeek/Kimi） | C1 的类脑实现 | 状态 |
|---|---|---|
| MLA 压缩 KV 缓存 | **O(1) 液态工作记忆——不是压缩缓存，是消除缓存** | ✅ 已验证（8063×）|
| MoE 参数稀疏 | **稀疏时间尺度路由——不是少用参数，是少算** | ✅ 已验证（+7.9% CPU）|
| MTP 多 token 预测 | **多尺度预测编码（k 步前瞻升级）** | 🟡 升级中 |
| GRPO 纯 RL 涌现推理 | **液态 GRPO——在液态动力学上做组相对策略优化** | 🔴 首个实验 |
| 系统1/系统2 双轨 | **DualSpeedSentry 快反射 + 慢思考（已有）** | ✅ 已验证 |

**核心差异（作为假设提出，非已证结论）**：DeepSeek/Kimi 在 transformer 框架内做工程优化；C1 探索的是"液态框架内架构重构"能否在长上下文+端侧成立——**注意**：M1 实测（hybrid 124.80 vs 纯液态 332.99）表明注意力是窗口内质量的基础，因此 C1 不以替代混合路线为目标；MoE 参数稀疏在 M3（混合版）中以 DeepSeekMoE 配方独立推进。

---

## 仓库结构

```
mt_lnn/                     # 自带的液态模型包（从 AwareLiquid/M1 vendor，见 VENDORED_M1.md）
benchmarks/
  reasoning_tasks.py        # parity 等合成推理任务（E1 用）
docs/
  ARCHITECTURE_C1.md        # 五层认知栈完整 spec
  STRATEGY_VS_DEEPSEEK_KIMI.md  # 迭代逻辑分析与启示
  EXPERIMENT_E1_LIQUID_GRPO.md  # 液态 GRPO（RL 涌现推理，对标 R1）
  EXPERIMENT_E2_MTP.md          # MTP 前瞻预测（对标 DeepSeek MTP）
  EXPERIMENT_E3_TEMPORAL_SCALING.md # 时间尺度扩容（液态版 MoE 实验）
  EXPERIMENT_E5_MULTI_HORIZON.md    # 多视野统一输出（EnergyTS 对标）
  EXPERIMENT_E6_ZERO_SHOT_TRANSFER.md # 跨域零样本迁移（EnergyTS 对标）
  ROADMAP_2B.md             # 实验路线图
  S1_RESULTS.md             # S1 结果（E1 GRPO / E2 MTP / E3 时间尺度）
  S2_RESULTS.md             # S2 结果（E3 已验证协议 / E2 长预算 / E1 热启 GRPO）
  S3_RESULTS.md             # S3 结果（E1 GRPO 稳定化 / E3 尺度确认）
  S4_RESULTS.md             # S4 结果（E5 多视野统一 / E6 跨域零样本）
  S5_RESULTS.md             # S5 结果（E5 6-seed 确认 / E6 5-seed 确认）
  S6_RESULTS.md             # S6 结果（E5 视野依赖 4/16/64/256）
  S7_RESULTS.md             # S7 结果（E5 视野边界修复 / 权重模式消融）
experiments/
  e1_liquid_grpo.py         # E1 可跑脚本（parity + GRPO + 长度外推评估）
  e2_mtp.py                 # E2 可跑脚本（baseline/mtp/state-mtp）
  e3_temporal_scaling.py    # E3 可跑脚本（n_scales x topk 矩阵）
  e5_multi_horizon.py       # E5 可跑脚本（unified/dedicated 多视野）
  e6_zero_shot_transfer.py  # E6 可跑脚本（留一域零样本迁移）
  prepare_data_e2.py        # E2 数据管线（WikiText-103 → uint16 .bin）
requirements.txt            # 独立运行依赖（无需 M1 checkout）
VENDORED_M1.md              # vendor 溯源（来源/commit/范围）
README.md
```

## 状态

- 🟡 **E1 液态 GRPO**：监督臂云 GPU 首跑完成——长度外推 acc 1.0 / 0.92 / 0.59 / 0.58 / 0.56（seed0，长度 32→128）；GRPO 臂进行中
- 🟡 E2（MTP 前瞻）/ E3（时间尺度扩容）：代码就绪 + 冒烟过 CI，待 GPU 全量
- 🟡 **E5 多视野统一 / E6 跨域零样本（EnergyTS 对标，2026-09 新增）**：设计完成 + 冒烟可跑——短↔长统一头、冷启能力边界
- 🟢 进行中：125M 收敛 + selective_decay A/B（DGX Spark）
- 借鉴基础：MT-LNN 已验证的 O(1) 内存 / 跨窗口记忆 / 稀疏共振 / 预测编码

---

*C1 = Cortex = 大脑皮层 = 高级认知的家。*
