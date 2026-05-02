
# Meta Prompt / PriorRes 项目当前状态说明，截至作完Caltech101的实验复现

## 1. 项目目标

本项目基于 CoOp/CLIP，构建一个 **dataset-conditioned prior residual prompt learning** 框架（简称 `CoOpPriorRes`），核心目标分为三层：

1. 在基础 few-shot 分类任务上，相比原始 CoOp 提升或至少逼近网格最优。
2. 通过引入 dataset-level prior（主线 `no_b`）提升跨数据集泛用性。
3. 通过引入 sample-level weighting（`b` 分支）探索在 base-to-novel（B2N）与 domain generalization（DG）任务上的潜在增益。

当前经验结论是：

- **主线 `USE_B=False`** 更稳定，是当前默认主模型。
- **`b` 分支 (`USE_B=True`)** 已经在多个数据集上接通，但不同 seed 的 `keff` 差异可能较大，说明其确实生效，但稳定性不如 `no_b`。
- 因此，当前推荐把 `b` 视为 **增强模块**，而不是默认主配置。

---

## 2. 当前主模型定义

### 方法名
`CoOpPriorRes`

### 当前主线默认配置
- backbone: `RN50`
- trainer: `CoOpPriorRes`
- config: `configs/trainers/CoOp/rn50_ep50.yaml`
- context position: `end`
- CSC: `False`
- 默认主线：
  - `USE_B=False`
  - `META_LR_RATIO=0.3`
- b 分支测试时：
  - `USE_B=True`
  - `B_LOSS_WEIGHT=0.2`
  - `META_LR_RATIO=0.3`

### 当前统一测试点
- `k = 16`
- `m = 16`
- `seed = 1, 2, 3`

### 当前结果命名规范
正式结果统一使用：
- `*_nob_fresh`：主线 no_b 正式结果
- `*_b0p2_fresh`：b 分支正式结果
- `*_fresh`：CoOp baseline 正式结果

排除项：
- `smoke`
- `cachebuild`

---

## 3. 当前项目结构中最关键的文件

### 3.1 Trainer / 主逻辑
- `third_party/CoOp_clean/trainers/coop_priorres.py`

### 3.2 Meta prompt / prior / b-weighting 相关
- `src/meta_prompts/prior_residual_adapter.py`
- `src/meta_prompts/shot_weighting.py`
- `src/meta_prompts/task_feature_extractor.py`
- `src/meta_prompts/task_feature_loader.py`

### 3.3 统一运行脚本
- `third_party/CoOp_clean/scripts/ours/run_priorres_any_dataset.sh`

### 3.4 CoOp baseline 网格脚本
- `run_coop_grid.py`

### 3.5 数据集 loader（尤其是已改过 b 分支的）
重点关注：
- `third_party/CoOp_clean/datasets/oxford_pets.py`
- `third_party/CoOp_clean/datasets/eurosat.py`
- `third_party/CoOp_clean/datasets/dtd.py`
- `third_party/CoOp_clean/datasets/food101.py`
- `third_party/CoOp_clean/datasets/oxford_flowers.py`
- `third_party/CoOp_clean/datasets/caltech101.py`

### 3.6 B2N / DG 参考脚本
- `third_party/CoOp_clean/scripts/cocoop/base2new_train.sh`
- `third_party/CoOp_clean/scripts/cocoop/base2new_test.sh`
- `third_party/CoOp_clean/scripts/cocoop/xd_train.sh`
- `third_party/CoOp_clean/scripts/cocoop/xd_test.sh`

---

## 4. 当前已跑数据集与状态

> 注：下表中的“已完成”表示至少已有 CoOp baseline + PriorRes 主线结果或相关代码已接通。

### 4.1 已跑基础任务数据集
- `oxford_pets`
- `eurosat`
- `dtd`
- `food101`
- `oxford_flowers`
- `caltech101`（命名冲突问题已解决）

### 4.2 当前经验总结
#### OxfordPets
- 基础任务表现较稳定。
- b 分支已接通并可运行。
- 适合作为 b 分支的较稳定参考集。

#### EuroSAT
- 高 domain gap。
- `no_b` 通常可以逼近或接近 CoOp 最优。
- `b` 分支已接通，但 seed 间波动较大。
- 经验上不建议把 EuroSAT 的 b 分支当作主卖点。

#### DTD
- 纹理数据集。
- 是验证 b 分支和主线泛用性的关键数据集之一。
- 已适合作为后续继续分析的重点数据集。

#### Food101
- 自然图像类别语义清晰。
- 比 EuroSAT 更适合作为稳定对照集。
- 可作为后续 B2N / DG 扩展前的稳定基础集。

#### Oxford Flowers
- 已完成下载与基础任务接通。
- 当前也用于统一命名风格和结果统计。

#### Caltech101
- 已完成基础任务与 b 分支接通。
- 数据真实目录名是 `caltech-101`。
- 配置命名冲突问题已经通过补充 yaml 文件名解决。
- 该问题已视为 **解决**，后续不要再把它当成未解决事项。

---

## 5. 当前关于 b 分支的结论

### 5.1 现象
开启 `USE_B=True` 后，不同 seed 下 `keff` 可能差异很大，例如出现：
- 某些 seed `keff ~ 14`
- 某些 seed `keff ~ 8`
- 某些 seed `keff ~ 1`

这说明：
- b 分支已经真实参与训练
- 它学到的是 **sample-level reweighting**
- 但其稳定性不如主线 `no_b`

### 5.2 当前结论
- `no_b` 是当前默认主配置。
- `b` 分支在基础任务上：
  - 可能提升某些 seed
  - 也可能使某些 seed 掉点
- 但在 **B2N** 和 **DG** 上，b 更可能有真正价值，因为它更贴近“样本可靠性重加权”的功能，而不是单纯提升 base 拟合。

### 5.3 当前推荐
后续新代码开发中：
- 先以 `no_b` 跑通 B2N / DG 主线
- 再在相同框架上添加 `b` 分支做最小对照

---

## 6. 数据与元文件说明

### 6.1 当前训练依赖的关键元文件
每个数据集都尽量保留：
- `split_zhou_*.json`
- `split_fewshot/shot_{k}-seed_{seed}.pkl`
- `split_fewshot/shot_{k}-seed_{seed}-slotproto.pkl`
- `outputs/task_features/<dataset>_train.json`

### 6.2 当前 task feature
主线依赖：
- `outputs/task_features/<dataset>_train.json`

如果新数据集没有该文件，需要先离线提取。

---

## 7. 当前统一运行方式

### 7.1 CoOp baseline
使用：
- `run_coop_grid.py`
- 或直接 `python train.py ... --trainer CoOp`

### 7.2 PriorRes 主线
统一入口：
- `third_party/CoOp_clean/scripts/ours/run_priorres_any_dataset.sh`

### 7.3 正式主线命令模板
```bash
for SEED in 1 2 3; do
  CUDA_VISIBLE_DEVICES=1 USE_B=False META_LR_RATIO=0.3 \
  OUT_DIR=/workspace/meta_prompt_1/third_party/CoOp_clean/output/<dataset>/CoOpPriorRes/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed${SEED}_nob_fresh \
  bash third_party/CoOp_clean/scripts/ours/run_priorres_any_dataset.sh <dataset> 16 16 ${SEED}
done