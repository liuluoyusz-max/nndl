# 成员3生成结果与审计记录

这里保存已完成运行的小体积交付物。完整原始数据、模型权重和增强图片不提交到 Git。

## 正式生成

`full_a0p5_b0p7/` 对应1349张 Photo Training 原图，每张生成一张结构保持增强图。
参数 alpha=0.5、beta=0.7，内容和风格短边512、JPEG质量95、NN CPU环境12线程。

- `manifest_ours.txt`：六列清单，路径相对于 DATA_ROOT，供成员4使用。
- `photo_train.txt`：两列清单，路径相对于 `DATA_ROOT/aug_sp/images/`，标签1～7。
- `pairs.txt`：生成图、内容图、风格名和标签的配对表。
- `generation.json`：代码、权重、输入/输出哈希、参数和完成记录。
- `delivery_check.json`：全量解码、标签、冻结配对、文件哈希和清单一致性检查结果。
- `structure_metrics.json`：成员4脚本计算的全量 EdgeScore，n=1349；LPIPS未运行。
- `analysis.json`：生成器的运行摘要；正式生成时未开启逐图指标，指标以 `structure_metrics.json` 为准。

这些清单用于审计和对接，读取图片前仍需要本地实际的 `data/aug_sp/images/`。
生成命令（从仓库根目录执行，先准备 PACS 和 Python 依赖）：

```powershell
python -m member3_structure_preserving.generate build --alpha 0.5 --beta 0.7 --threads 12
```

成员4使用生成后的 `data/augmented/manifest_ours.txt`；该清单与
`data/aug_sp/manifest_ours.txt` 内容一致，不重复保存图片。

## 样例与预实验

- `sample_index.json`：七类真实内容图与 `../figures/*_comparison.png` 的对应关系。
- `beta_sweep_7.json`：beta={0,0.3,0.5,0.7,1} 的7张 dog 图片预实验，非全类别消融。
- `pacs_source.json`：PACS下载来源、压缩包大小和哈希、原图数量。

JSON中的本机绝对路径已转成仓库相对路径。导出记录统一使用LF换行；
`delivery_check.json` 同时保存原始与导出后的 generation记录哈希。
图像、输入文件、模型权重及代码哈希保持原始审计含义。跨平台算子和环境版本可能影响逐像素复现，
以 `generation.json` 中记录的环境为参照。

beta=0.7是初始设置，未根据目标域测试结果选优。本次交付未运行分类训练，
这些结构指标不能代表跨域分类准确率的改善。
