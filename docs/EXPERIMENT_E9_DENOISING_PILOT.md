# E9 — 类脑数据形态试点：加噪→去噪监督（换掉 next-token）

> 状态：🟡 设计完成（冒烟可跑，待 GPU 全量）
> 对标：**liquid-research-skill 数据形态层**（"数据形态 = 拟合目标；换架构不同步换数据形态 =
> 只换了一半范式"）+ skill 液态落地建议 #3（"在液态状态演化环节试点类扩散逐步修正"）
> 核心问题：**把监督信号从"预测下一个 token"换成"加噪→去噪逐步修正"，液态栈的长度外推是否提升？**

---

## 1. 动机

skill 的诊断（data-form.md §1）直指我们当前的研发困境：
> "连续时间动力学在 next-token 文本上无处演化，归纳偏置被浪费。"
> "扩散模型有效，是因为它的数据形态是'噪声→清晰'对、监督是'逐步去噪'，与归纳偏置绑定。"

我们最近的实证恰好印证：E7 判负（事件边界形态对语言线无受力面）、E1/E2/E3 全负、
B-1 DEAD——**所有 next-token 变体的架构改动都只回落到量变**（skill 结论 2）。

E9 = skill 推荐的最小范式探针：**同一个液态模型、同一个任务域（parity，已验证可学），
只换监督信号**——输入加噪，训练目标是重建干净序列（逐步修正），而不是预测下一个 token。

## 2. 假设

**H-E9**：用"加噪→去噪"监督训练的液态模型，在**长度外推**（训练 L≤32 → 测试 L=48~128）
上显著优于同预算 next-token 监督的模型（3-seed 配对）；且 K 步逐步修正 ≥ 单步去噪
（"逐步"这一扩散要素有增量）。

## 3. 实验设计

### 任务：parity d16（已验证可学域，避开 budget wall——ADJ-001 教训）

### 三种监督信号（同模型、同预算、同任务）

| 模式 | 输入 | 监督信号 | 对应 |
|---|---|---|---|
| `next_token` | 干净序列 | 预测下一个 token（CE） | 现状（transformer 标准） |
| `denoise` | **加噪序列**（位翻转 p=0.15） | 预测**干净**序列（全位置 CE） | 预测编码：加噪→去噪对 |
| `progressive` | 加噪序列 | **K=3 步逐步修正**：每步输入=上一步 argmax 输出，每步对干净目标 CE | 类扩散：逐步去噪 |

### 指标

1. **长度外推**：训练 L∈[8,32]，测试 L∈{48, 64, 96, 128} 的准确率（液态栈的招牌优势）
2. **抗噪鲁棒性**：测试时注入噪声（p=0.05~0.3），denoise/progressive 是否比 next_token 更稳

## 4. 预注册判据（跑之前写死）

- **H-E9 成立** ⇔ denoise 或 progressive 在 ≥2 个外推长度上显著优于 next_token（3-seed
  配对 sign-test p<0.05），且 progressive 在 ≥1 个长度上优于 denoise（"逐步"增量存在）；
- **任一不成立** → 如实记 DEAD（"去噪监督无外推增量"），不事后移动判据；
- **诚实边界**：E9 是 skill 明示的"归纳偏置借用"（"treat it as an inductive-bias borrowing,
  not an architecture swap"）——小规模验证，不迁移整套扩散训练方法。

## 5. 产出

- `experiments/e9_denoising_pilot.py`：可跑脚本（`--mode next_token/denoise/progressive`、`--smoke`）
- 结果 JSON：3 模式 × 3 seeds 的外推曲线 + 抗噪曲线
- 判定：监督信号是否真的比架构改动更能撬动液态栈的招牌优势

## 6. 依赖与边界

```bash
# 合成冒烟（CPU，零数据文件）
python experiments/e9_denoising_pilot.py --smoke --mode denoise --steps 3 --seeds 0
# 全量（GPU 队列，排在 B-2 tau 之后）
python experiments/e9_denoising_pilot.py --mode denoise --steps 6000 --seeds 0 1 2
```

- **与 E5/E7 的关系**：E5 测"输出侧视野"、E7 测"输入侧形态"（均判负/噪声），E9 测
  "监督信号"——data-form.md 的三个维度逐一验证。
- **parity 只是最小实例**：skill 的 data-form 表还列了状态转移轨迹（前向模型）、
  episode（情景记忆）、RPE（动作选择）——E9 只验证"去噪监督"这一种，其余按组件立项
  （见 docs/COMPONENT_BACKLOG.md）。
