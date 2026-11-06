# 成员4 交接文档：训练/评估框架使用说明

> 给成员2、成员3的一次性对接说明（按分工约定，各收一次配置说明，之后不互相等待）。

## 1. 你们需要交付什么

**给成员2（AdaIN）和成员3（结构保持）各一份：**

1. **增强图片目录**：图片写到 `data_root/augmented/<group>/` 下
   （`<group>` = `adain` 或 `ours`）。任何文件名都行，只要在 manifest 里写对路径。
2. **manifest 文本文件**（核心交付物，训练只认这个）：
   `data_root/augmented/manifest_<group>.txt`，每行格式：

   ```
   <增强图片相对data_root的路径> <标签1..7> <原图相对路径> <风格名> <alpha> [beta]
   augmented/adain/056_0022__scene_de_rue.jpg 1 photo/dog/056_0022.jpg scene_de_rue 0.5
   augmented/ours/056_0022__scene_de_rue.jpg 1 photo/dog/056_0022.jpg scene_de_rue 0.5 0.7
   ```

   - 前两列与成员1的 split 格式一致（标签是 1..7，训练代码内部减1）；
   - `原图相对路径` 必须是 `photo_train.txt` 里出现的路径（不带 `PACS/` 前缀）；
   - `风格名` 必须是 style pool 里的名字（见第3节）；
   - ours 的行多一个 `beta` 列。

   **成员2注意**：你现在的 `build_adain_aug.py` 写出的是两列
   （`路径 标签`）的 `photo_train.txt`，放在你自己的 `data/aug_adain/` 目录下，
   训练代码读不了。请改成上面这个五列格式，图片写到
   `data_root/augmented/adain/`，manifest 写到
   `data_root/augmented/manifest_adain.txt`。
   文件名建议 `<原图stem>__<风格名>.jpg`，例如
   `augmented/adain/056_0022__scene_de_rue.jpg`。
   另外你现在的 `output_relative()` 在 `styles_per_image=1` 时会直接返回
   **原图路径**（`photo/dog/056_0022.jpg`），会和原图撞名，必须改掉。

   **懒得改生成器也行**：我写了一个转换脚本，喂给它你现有的
   `pairs.txt`（tab 分隔：输出路径 / 原图 / 风格图 / 标签）就行，
   它会转格式、搬图片、核对配对表：

   ```bash
   python convert_aug_manifest.py --group adain \
       --pairs <你的输出目录>/pairs.txt --alpha 0.5
   ```

   **alpha 注意**：你的 `build_adain_aug.py` 和 `generate_adain_samples.py`
   里 `--alpha` 的默认值是 **1.0**，但本项目的冻结协议是 **0.5**
   （3.4 / 3.7 都按 0.5 写，实验 4 再扫 0.25/0.5/0.75/1.0）。
   生成正式数据集时请显式传 `--alpha 0.5`，manifest 里的 alpha 列也要写 0.5。
   你现有的 8 张样例是 alpha=1.0 生成的，重做样例时也请改成 0.5。

## 2. 三组公平比较的硬约束（fairness_check.py 会机器校验）

- 每张 `photo_train` 原图生成 **恰好 1 张** 增强图（1349张原图 → 1349张增强图）；
- 两组使用**同一套风格图**、**同样的 原图→风格 配对**（配对表已冻结，见下）；
- 两组的 manifest 行数相同（1349）；
- 生成用 alpha 默认 0.5（实验4再扫 0.25/0.5/0.75/1.0，届时各出一份 manifest）；
- **风格图不能是 PACS 的图**（见第3节），checker 会逐张比对内容哈希。

校验命令（两组 manifest 都到齐后）：

```bash
python fairness_check.py --adain <adain manifest> --ours <ours manifest> \
    --runs ../runs/original_s0 ../runs/original_s1 ../runs/original_s2
```

校验不过的 manifest 不会进入训练。交付前请自查一遍。

## 3. 冻结的风格图池与配对（已生成，双方直接用）

`code/style_pool/style_pool.json` 包含：

- **10 张风格图**（**全部来自 PACS 之外**的外部画作，文件名即风格名，
  图片在 `code/style_pool/images/`，每张记录了 sha256）；
- **原图→风格配对表**（seed 42 冻结）：photo_train 每张原图对应哪张风格图。

> **⚠️ 重要：风格图必须来自 PACS 之外。**
> Art Painting / Cartoon / Sketch 是本项目的**测试域**。
> 如果风格图取自这三个域，就等于把测试集的外观送进了训练集，
> 跨域准确率会虚高，实验结论不成立。
> 早先的版本曾经用 PACS art_painting 的 16 张图当风格池，
> 16 张**全部落在 art_painting_test.txt 里**，属于测试集泄露，已经作废。
> 现在冻结的 10 张是外部画作（AdaIN 官方 demo 风格集，来自 WikiArt），
> `build_style_pool.py` 和 `fairness_check.py` 都会用内容哈希
> 与本地 PACS 逐一比对，发现任一风格图就是 PACS 图就直接报错。

**风格名**（manifest 第 4 列必须用这些名字之一）：

```
asheville   brushstrokes  contrast_of_forms  flower_of_life  la_muse
mondrian    picasso_self_portrait           scene_de_rue
the_resevoir_at_poitiers                   woman_with_hat_matisse
```

**配对规则**（与成员2的 `StylePool.assign` 完全一致）：
把上面 10 个文件名按字母序排好，用 `random.Random(42).shuffle` 打乱，
再沿打乱后的顺序循环取用。所以第 i 张 photo_train 原图
（按 `photo_train.txt` 的行序）的风格是 `order[i % 10]`。
成员3 直接读 `style_pool.json` 里的 `pairing` 字段即可，不要自己重算。

成员2如对风格图池有不同选择，**在生成前**告诉我，
我重新冻结配对表；生成之后不可换。风格图池数量默认10，可商量。

## 4. 我的训练框架（已验证可跑）

- `config.py`：全部共享超参数（预算 660 步、batch 64、SGD、cosine、label smoothing 0.1、
  训练/测试 transform）。**不要**在自己的代码里改这些；三组共用一份。
- `train.py` / `run_all.py` / `evaluate.py` / `summarize_results.py`：
  统一训练（step-budget 制，三组同优化量）、多 seed、评估、汇总出表。
- `verify_splits.py`：数据校验（已通过：六份 split 的数量/类别分布/sha256 与
  protocol.json 完全一致，9990 张全部在盘）。

环境：本机已装 CUDA torch 2.11（RTX 4070 Laptop 8GB），单次训练约 10-12 分钟。

## 5. 冒烟测试用的小样例（可直接模仿）

`make_synthetic_aug.py` 生成了一份 40 张的假增强集（恒等"风格化"）+ 两份 manifest，
用于测试训练管线和 fairness checker。你们写真实生成器时，
manifest 格式照抄 `data/augmented/manifest_adain_synthetic.txt` 即可。

## 6. 时间安排（相对进度报告 +4 周）

- 第1周内请交付：风格池确认/替换意见（成员2）+ 两组 manifest 与图片（成员2、3）；
- 第2周我跑实验2、3；第3周实验4扫 alpha/beta；第4周补 seed + 汇总出表。
