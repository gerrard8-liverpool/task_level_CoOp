# Meta Prompt 1 项目 README（对话接力版）

## 1. 这个文档的用途

这个 README 的目标不是给陌生人做完整论文介绍，而是给 **新的 ChatGPT 对话 / 新协作者 / 未来的我自己** 快速建立项目上下文。

标准是：

- 看完后能理解项目在做什么；
- 知道当前代码结构和关键文件在哪里；
- 知道哪些实验已经完成，哪些还在推进；
- 知道当前最稳的运行方式；
- 知道接下来应该优先帮我做什么，而不是重复走弯路。

---

## 2. 项目一句话概述

本项目是在 **CoOp（Context Optimization for CLIP prompts）** 基础上，引入 **dataset-conditioned prior + residual joint adaptation** 的架构：

- 先从数据集级统计特征中提取任务特征；
- 通过一个轻量元网络输出任务级先验参数 `a` 和 `b`；
- `a` 用于控制有效 prompt capacity；
- `b` 用于控制固定训练预算下不同 support slot 的利用权重；
- 然后与 CoOp 进行端到端联合训练。

当前定位不是“完全动态重写 prompt learner”，而是：

> **让数据集统计特征决定 prompt 相关的任务级先验，再通过小残差进行稳定修正。**

---

## 3. 核心研究问题

### 3.1 a 分支要回答什么

给定一个数据集，是否可以不依赖人工离散网格搜索，而是直接由数据集统计特征生成一个更适合该任务的 **prompt capacity prior**？

这里的核心观测量是：

- `a = (a1, ..., a16)`
- `meff = sum(a)`

`meff` 表示结构上仍保留 `n_ctx=16`，但功能上有效利用了多少上下文容量。

### 3.2 b 分支要回答什么

在固定训练预算（例如每类统一 16-shot）的前提下，是否可以不把 16 个 support 样本都等价对待，而是由数据集特征生成一个 **training utilization prior**，让模型自动学习不同 slot 的利用方式？

这里的核心观测量是：

- `slot_weights = softmax(b_logits)`
- `keff = 1 / sum(slot_weights^2)`
- `b_entropy`
- `top1_weight`
- `top4_weight_sum`

当前理解里：

- `keff` 越接近 16，说明越接近均匀利用；
- `keff` 越低，说明权重越集中；
- `b_entropy` 越低，说明分布越尖；
- `top1_weight / top4_weight_sum` 越高，说明更偏向少数典型 slot。

---

## 4. 当前项目阶段

项目目前已经完成两段最重要的早期推进。

### 4.1 第一阶段：OxfordPets 上的 a 分支主验证（已完成）

已经完成基于 CoOp 的 dataset-only prior 主版本 few-shot sweep，核心结论是：

- 模型能够稳定接入 CoOp；
- 相比早期“强门控直接相乘”方案，当前 residual 版本训练稳定；
- 在不同 shot 下，模型学到的有效容量 `meff` 基本稳定在约 `11.x` 左右；
- 说明模型更像是在学习 **任务级固定先验容量**，而不是为每个 shot 拟合一个最优 m；
- 相比 CoOp 默认配置已经有明显收益；
- 相比网格寻优最优结果也已经基本逼近。

当前对第一阶段最稳妥的理解是：

> **dataset-conditioned prior 是主导项，residual 是小修正。**

### 4.2 第二阶段：固定 16-shot 下的 b 分支接入（已打通）

第二阶段已经完成最关键的一步：

- `b` 分支已经成功接入端到端训练；
- slot 不再动态构造，而是通过离线预处理固定；
- support slot 的语义由“类内样本到 prototype 的距离排序”定义；
- 初始版本中，`b` 先经过 sigmoid 再 softmax，导致分布过平；
- 后来已改为 **直接对 `b_logits` 做 softmax**，非均匀利用分布开始变得更明显。

当前结论是：

> **b 分支的机制验证已经成立，但其在 OxfordPets 16-shot 上的最终精度收益还不显著。**

所以目前对 b 的定位不是“已经带来明显涨点”，而是：

- 已能训练；
- 已能学出非均匀利用分布；
- 更适合在多数据集和 DG 场景中进一步验证价值。

---

## 5. 下一步主线（重要）

当前项目的优先级已经明确：

### 5.1 先做多数据集验证

优先目标不是继续在 OxfordPets 上打磨消融，而是回答：

- 不同数据集是否会学到不同的 `a / meff`？
- 不同数据集是否会学到不同的 `b / keff / entropy`？
- 当前方法是否真的是 **dataset-conditioned prior**，而不是 OxfordPets 偶然有效？

### 5.2 然后进入 DG

DG 才是 b 分支更自然、更强的发挥场景。

在 DG 中，b 的语义会从 few-shot support weighting 转变为：

- source-domain utilization prior
- domain weighting prior

这比继续在单一数据集上抠 0.1 个点更符合方法原始叙事。

### 5.3 消融放后面

当前不是不做消融，而是 **不优先做**。

原因：

- 目前更需要先把“跨数据集成立”做出来；
- 再做 DG，强化方法价值；
- 真正严格的结构消融可以后置。

---

## 6. 项目目录结构（关键）

### 6.1 顶层结构

项目工作根目录（容器内）通常是：

```text
/workspace/meta_prompt_1
```

其中最重要的几个目录：

```text
/workspace/meta_prompt_1
├── src/                         # 我的元学习/任务特征/加权相关代码
├── scripts/                     # 我自己写的运行脚本和辅助脚本
├── outputs/                     # 任务特征等输出文件
├── third_party/
│   ├── CoOp_clean/              # 当前真正使用的 CoOp 改造版本
│   ├── CoOp/                    # 旧版或原始路径（不一定是当前主分支）
│   └── Dassl.pytorch/           # CoOp 依赖的数据/训练框架
└── ...
```

### 6.2 当前最重要的代码位置

#### A. Trainer 主文件

```text
/workspace/meta_prompt_1/third_party/CoOp_clean/trainers/coop_priorres.py
```

这是当前最关键的训练逻辑文件，负责：

- prompt learner + prior adapter 的联合训练；
- warm-up / ramp-up / alternating optimization；
- `a` 和 `b` 的 loss 接入；
- 日志里 `meff / keff / entropy / top1 / top4` 等统计项。

#### B. 元学习适配器

```text
/workspace/meta_prompt_1/src/meta_prompts/prior_residual_adapter.py
```

负责：

- 读取数据集任务特征；
- 调用元网络输出 `a0 / b0`；
- 通过 `delta_a / delta_b` 构造残差；
- 输出 `a / b / b_logits / meff / keff / lambda_t` 等。

#### C. b 分支加权损失

```text
/workspace/meta_prompt_1/src/meta_prompts/shot_weighting.py
```

负责：

- 从 batch 中读取 `slot_id`；
- 根据 `softmax(b_logits)` 构造 `slot_weights`；
- 计算 weighted few-shot loss；
- 输出 `keff_from_weights / entropy / top1 / top4` 等辅助统计。

#### D. 数据集定义（当前最先适配的是 OxfordPets）

```text
/workspace/meta_prompt_1/third_party/CoOp_clean/datasets/oxford_pets.py
```

负责：

- 数据集读取；
- few-shot pkl 缓存加载；
- 在 `USE_B=True` 且 `NUM_SHOTS=16` 时，要求加载 `slotproto` 版本缓存。

#### E. Dassl 数据返回接口

```text
/workspace/meta_prompt_1/third_party/Dassl.pytorch/dassl/data/data_manager.py
/workspace/meta_prompt_1/third_party/Dassl.pytorch/dassl/data/datasets/base_dataset.py
```

当前已改造点：

- `Datum` 支持 `slot_id / slot_rank / dist_to_proto`；
- `DatasetWrapper.__getitem__()` 会返回这些字段；
- trainer 因此可以在 batch 中拿到 `slot_id`。

---

## 7. 关键脚本

### 7.1 生成 slot-aware few-shot cache

```text
/workspace/meta_prompt_1/scripts/ours/build_slotproto_cache_oxfordpets.py
```

用途：

- 读取普通 `shot_16-seed_X.pkl`；
- 提取冻结 CLIP 图像特征；
- 对每类 support 求 prototype；
- 按到 prototype 的距离排序；
- 为每个 `Datum` 生成 `slot_id / slot_rank / dist_to_proto`；
- 输出 `shot_16-seed_X-slotproto.pkl`。

### 7.2 第二阶段 16-shot 训练脚本

```text
/workspace/meta_prompt_1/scripts/ours/oxfordpets_priorres_stage2_16shot.sh
```

支持三种模式：

- `aonly`
- `bonly`
- `ab`

### 7.3 第二阶段汇总脚本

```text
/workspace/meta_prompt_1/scripts/ours/summarize_oxfordpets_stage2.py
/workspace/meta_prompt_1/scripts/ours/export_oxfordpets_stage2_csv.py
```

用途：

- 汇总各个 seed 的指标；
- 导出 CSV；
- 用于后续整理表格。

---

## 8. 输出与缓存路径（非常重要）

### 8.1 few-shot 缓存

普通 few-shot 缓存：

```text
/workspace/datasets/oxford_pets/split_fewshot/shot_16-seed_1.pkl
```

带 slot 语义的缓存：

```text
/workspace/datasets/oxford_pets/split_fewshot/shot_16-seed_1-slotproto.pkl
```

当 `USE_B=True` 且 `DATASET.NUM_SHOTS=16` 时，代码会强制要求后者存在。

### 8.2 task feature 文件

```text
/workspace/meta_prompt_1/outputs/task_features/oxford_pets_train.json
```

目前 task feature JSON 中包括：

- 数据集名、split、backbone；
- `raw_features`；
- `transformed_features`；
- 类别数、类名、样本数等。

### 8.3 训练输出目录

CoOp baseline：

```text
/workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_pets/CoOp/rn50_16shots_seed1
```

Stage-2 PriorRes：

```text
/workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_pets/CoOpPriorRes_stage2/aonly/rn50_16shots_beta0.2_seed1
/workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_pets/CoOpPriorRes_stage2/bonly/rn50_16shots_beta0.2_seed1
/workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_pets/CoOpPriorRes_stage2/ab/rn50_16shots_beta0.2_seed1
```

---

## 9. 当前训练协议（默认理解）

### 9.1 第一阶段（旧主验证）

- backbone: RN50
- n_ctx: 16
- CSC: False
- class token position: end
- dataset-only prior
- warm-up: 5
- ramp-up: 10
- alternating optimization
- meta lr ratio: 0.3

### 9.2 第二阶段（固定预算）

- `DATASET.NUM_SHOTS = 16`
- `Kmax = 16`
- `slot_id` 使用离线排序结果
- `slot_weights = softmax(b_logits)`
- weighted loss 只在 `USE_B=True` 时启用
- `β = TRAINER.COOP.B_LOSS_WEIGHT`，当前常用值为 `0.2`

### 9.3 当前三种模式含义

- `aonly`：只启用 a 分支，b 不参与 loss
- `bonly`：关闭 context gating，只让 b 控制 training utilization
- `ab`：a 和 b 同时启用

注意：

当前这三者更接近 **branch usage ablation**，不是最严格意义上的“true structural ablation”。

---

## 10. 当前阶段已经形成的认知

### 10.1 已经比较确定的事

1. `a` 分支是当前最稳定、最成熟的部分；
2. `dataset-conditioned prior` 是方法的核心主导项；
3. residual 当前主要是小修正，而不是大幅重写；
4. `b` 分支已经能学、能动、能形成非均匀分布；
5. 但 `b` 在 OxfordPets 16-shot 上的最终精度收益仍不显著；
6. 当前最该做的是多数据集验证，然后进入 DG。

### 10.2 当前不该误判的事

1. 不能因为 `aonly / bonly / ab` 差距小，就认为 b 没用；
2. 当前差距小，很大程度上因为它们共享 dataset prior，且 residual 被设计成小修正；
3. 当前第二阶段的主要价值是“机制验证”，不是“已经显著涨点”。

---

## 11. 当前最可能需要修改的文件

如果后续要继续改代码，优先级通常是：

1. `third_party/CoOp_clean/trainers/coop_priorres.py`
2. `src/meta_prompts/prior_residual_adapter.py`
3. `src/meta_prompts/shot_weighting.py`
4. `third_party/CoOp_clean/datasets/<dataset>.py`
5. `scripts/ours/*.sh`
6. `scripts/ours/*.py`

如果是把当前方法推广到新数据集，通常最先要做的是：

- 添加或检查 `datasets/<new_dataset>.py`；
- 生成该数据集的 task feature JSON；
- 检查 few-shot split 缓存路径；
- 复制/改写对应训练脚本。

---

## 12. 常见坑

### 12.1 `USE_B=True` 但 slotproto 缓存缺失

报错典型形式：

```text
Missing slot-annotated few-shot cache: ... shot_16-seed_X-slotproto.pkl
```

说明：

- 先跑了 `bonly/ab`
- 但没先生成对应 seed 的 slot-aware 预处理缓存

解决：

- 先运行 `build_slotproto_cache_oxfordpets.py`

### 12.2 重跑实验但日志混旧内容

原因：

- 没清空输出目录
- 新旧实验共用同一个 output path

解决：

- 先把旧目录归档或删除，再重跑

### 12.3 b 分支虽然训练了，但分布几乎均匀

这是之前出现过的问题。

原因：

- 旧版本先 `sigmoid(b_logits)` 再 `softmax(b)`，会把动态范围压平

已修复方案：

- 直接 `softmax(b_logits)` 生成 slot 权重

---

## 13. 给新对话助手的最小上下文（可直接复制）

如果我要在一个新的 ChatGPT 对话里快速接上项目，可以直接给对方这段：

```text
这是一个基于 CoOp 的科研项目。我在做 dataset-conditioned prior + residual joint adaptation。

核心是：
1. 从数据集级统计特征中提取 task feature；
2. 通过元网络输出任务级先验 a 和 b；
3. a 控制 prompt capacity（meff=sum(a)）；
4. b 控制固定 16-shot 预算下不同 support slot 的利用分布；
5. 当前 b 的权重是 softmax(b_logits)，而不是 softmax(sigmoid(b))。

当前代码主目录：
- /workspace/meta_prompt_1/src/meta_prompts/prior_residual_adapter.py
- /workspace/meta_prompt_1/src/meta_prompts/shot_weighting.py
- /workspace/meta_prompt_1/third_party/CoOp_clean/trainers/coop_priorres.py
- /workspace/meta_prompt_1/third_party/CoOp_clean/datasets/oxford_pets.py
- /workspace/meta_prompt_1/third_party/Dassl.pytorch/dassl/data/data_manager.py
- /workspace/meta_prompt_1/third_party/Dassl.pytorch/dassl/data/datasets/base_dataset.py

当前已经完成：
- OxfordPets 上的第一阶段 few-shot 主验证；
- a 分支已稳定成立；
- b 分支已接通并能学出非均匀 slot 分布；
- 但 b 在 OxfordPets 16-shot 上精度收益还不显著。

下一步优先做：
- 多数据集验证
- 然后做 DG
- 消融后置

如果你要帮我写代码，请先帮我理解当前目录和脚本，不要默认从零重新设计框架。
```

---

## 14. 推荐的后续协作方式

如果后续继续和 AI 协作，最好按下面方式提问：

### 14.1 如果要改代码

给出：

- 当前目标（例如：推广到 `food101`）
- 要改的文件路径
- 关键日志/报错
- 当前脚本命令

### 14.2 如果要分析实验

给出：

- CSV 或关键表格
- 对应 mode / seed / dataset
- 当前最想回答的问题（例如：是不是 prior 起主导作用）

### 14.3 如果要写论文

先明确当前写的是：

- 项目概述
- 实验设计
- 阶段性进展
- 结果分析
- 方法章节

这样更容易得到可直接使用的文字。

---

## 15. 当前一句话总结

> 这是一个以 CoOp 为基础、以 dataset-conditioned prior 为核心、当前已完成单数据集主验证并正在向多数据集与 DG 扩展的科研项目；a 分支已稳定成立，b 分支机制已接通，下一步重点不是继续抠单数据集消融，而是验证跨数据集与更强泛化场景中的方法价值。
