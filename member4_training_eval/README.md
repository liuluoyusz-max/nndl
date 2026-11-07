# Member 4 — Training / Evaluation / Ablation Code

> 统一训练与评估脚本。成员4负责，三组实验（Original / +AdaIN / +Ours）共用同一套代码，
> 唯一差别是读入的 manifest 不同。

## 0. 环境

```
torch 2.11.0+cu128   (RTX 4070 Laptop 8GB)
torchvision 0.26.0
numpy / scipy / opencv-python / pillow / tqdm
pip install lpips     # 可选，3.6 的 LPIPS 指标需要；缺失时脚本会自动降级并提示
```

数据根目录在 `config.py` 里配置：

```python
DATA_ROOT   = PROJECT_ROOT / "data"  # 可用 NNDL_DATA_ROOT 环境变量覆盖
IMAGE_ROOT  = DATA_ROOT / "PACS"      # 原始图片
AUG_ROOT    = DATA_ROOT / "augmented"  # 风格化图片（成员2/3 产出）
```

## 1. 文件清单

| 文件 | 作用 |
|---|---|
| `config.py` | **唯一真源**：三组共用的超参、split 路径、transform。`config_hash()` 用于公平性校验 |
| `model.py` | ResNet-18 构造 + 随机种子设置 |
| `data.py` | Dataset：原始图片与风格化图片走同一套代码路径 |
| `train.py` | 统一训练脚本（step-budget 制） |
| `evaluate.py` | 在六个 split 上评估，输出 JSON |
| `run_all.py` | 一个配置跑多 seed + 汇总 |
| `summarize_results.py` | 生成统一结果表（plain / markdown / latex 三种格式） |
| `fairness_check.py` | **公平性机器校验**：两组 manifest 是否满足 3.7 的全部约束 |
| `build_style_pool.py` | 冻结风格图池 + 冻结 原图→风格 配对（**含防泄露检查**） |
| `structure_metrics.py` | 3.6 的结构指标：LPIPS + EdgeScore |
| `verify_splits.py` | 数据校验：六个 split 对本地 PACS 的一致性检查 |
| `make_synthetic_aug.py` | 造 40 张假增强集，用于打通流程（**不可用于真实结果**） |
| `convert_aug_manifest.py` | 把成员2/3 的生成结果转成训练用的 manifest，并核对配对表 |
| `_sync_sections.py` | 由 `sections/member4_*.tex` 生成 `sections/member4_sections.tex` |
| `project_splits/` | 成员1 的六份清单（只读，不修改） |
| `style_pool/images/` | **10 张外部风格图**（不属于 PACS） |
| `style_pool/style_pool.json` | 冻结的风格图池 + 配对表 |
| `style_pool/SOURCES.txt` | 风格图来源与排除规则 |
| `GENERATOR_INTERFACE.md` | 给成员2、成员3 的一次性对接说明 |

## 2. 关键协议（已写入报告 3.3 / 3.7）

- **step-budget 制**：三组都是 660 个 optimizer step，与数据集大小无关。
  两个增强组图片数翻倍 → 对数据的遍历次数相应变少，保证优化预算一致。
- **不使用 checkpoint 选择**：报告固定预算下的**最后一个** checkpoint。
  原因见报告 3.3：Photo Validation 只有 150 张、60 步内就饱和到 97–98%，
  按 val 选点等于选噪声，把跨域准确率的 seed 间方差从 2.71 抬到 7.22。
- **不使用光度增强**：训练 transform 只用几何增强（RandomResizedCrop + flip）。
  color jitter / grayscale 会改变风格统计量，而那正是被研究的变量。
- **风格图池在 PACS 之外**：Art/Cartoon/Sketch 是**测试域**，风格图若取自其中
  就等于把测试外观送进训练。冻结的 10 张风格图全部是外部画作，
  `build_style_pool.py` 与 `fairness_check.py` 都会用内容哈希与本地 PACS 逐一比对，
  发现任一风格图就是 PACS 图就直接报错。
- **同 seed 集**：{0, 1, 2}。

## 3. 常用命令

```bash
# 0) 校验数据（已通过：9990 张，数量/类别/sha256 全对）
python verify_splits.py

# 1) 冻结风格池（只需跑一次；换风格图要在生成前重跑并通知成员2/3）
python build_style_pool.py --n 10 --alpha 0.5

# 2) 跑一组实验（3 seeds）
python run_all.py --group original
python run_all.py --group adain --aug-manifest <manifest>
python run_all.py --group ours  --aug-manifest <manifest>

# 3) 公平性校验（两组 manifest 齐了之后）
#    --runs 可选，用来核对各组 run 的训练配置指纹是否一致
python fairness_check.py --adain <adain manifest> --ours <ours manifest> \
    --runs ../runs/original_s0 ../runs/original_s1 ../runs/original_s2

# 4) 结构指标
python structure_metrics.py --manifest <manifest>

# 5) 出表
python summarize_results.py --format latex

# 6) 重新合并 3.1/3.3/3.6/3.7/5.2（改的是 sections/member4_*.tex）
python _sync_sections.py

# 7) 把成员2/3 的生成结果转成训练用的 manifest
#    （格式不对时用这个脚本转，它会核对配对表和风格池）
python convert_aug_manifest.py --group adain \
    --pairs <生成结果目录>/pairs.txt --alpha 0.5
python convert_aug_manifest.py --group ours \
    --pairs <生成结果目录>/pairs.txt --alpha 0.5 --beta 0.7
```

## 4. manifest 格式（成员2/3 交付物）

```
<增强图片相对 DATA_ROOT 的路径> <标签1..7> <原图相对路径> <风格名> <alpha> [beta]
augmented/adain/056_0022__scene_de_rue.jpg 1 photo/dog/056_0022.jpg scene_de_rue 0.5
augmented/ours/056_0022__scene_de_rue.jpg 1 photo/dog/056_0022.jpg scene_de_rue 0.5 0.7
```

风格名必须是 `style_pool/images/` 里的 10 个外部画作之一（不带扩展名）。
原图相对路径必须是 `photo_train.txt` 里出现的路径（不带 `PACS/` 前缀）。

## 5. 当前结果（2026-11-06）

Original 组，3 seeds，660 steps，最终 checkpoint：

| | Photo | Art | Cartoon | Sketch | Avg |
|---|---|---|---|---|---|
| Original | 99.0±0.3 | 63.2±3.7 | 25.1±1.8 | 31.5±4.2 | **39.9±2.7** |

实验 1（domain gap）已完成。实验 2/3 等两组 manifest 到位后启动。

## 6. 目录说明

- `runs/<tag>/` — 每个 run 的 checkpoint、训练历史、评估 JSON
- `runs/summary_<group>.json` — 多 seed 汇总
- `runs/_archive_bestval/` — 旧的 best-val 选点结果（保留作对比，说明为什么弃用）
