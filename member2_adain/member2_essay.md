# 成员2报告稿

阅读这一份。排版代码在 `member2_essay.tex`，参考文献数据库在 `refs.bib`。两边使用相同的章节编号。

| 报告位置 | 本文章节 | 排版代码位置 |
|---|---|---|
| 2.2 Style Transfer and AdaIN | 第一节 | `member2_essay.tex` 的 2.2 |
| 2.3 Style Augmentation for Image Classification | 第二节 | `member2_essay.tex` 的 2.3 |
| 3.4 AdaIN Style Augmentation | 第三节 | `member2_essay.tex` 的 3.4 |
| 4.2 AdaIN Augmentation Progress | 第四节 | `member2_essay.tex` 的 4.2 |
| References | 第五节 | `refs.bib` |

## 任务范围

实现只做普通 AdaIN。

- 调用接口是 `stylize(content, style, alpha)`。成员3另外使用 `encode`、`mix`、`decode`。
- 风格图是 10 张外部画作。随机种子是 42。只处理 Photo 训练集的 1349 张。
- 强度有四档：0.25、0.5、0.75、1.0。三组实验里的主比较使用 1.0。
- 已经完成 8 张内容图的 32 张预览。1349 张全量增强图还没有生成。

报告只写 2.2、2.3、3.4，并给成员5提供 4.2 的三条进度。结构图、参数 β、分类器训练和准确率表由其他成员写。

## 第一节　2.2 Style Transfer and AdaIN

这一节说明 AdaIN 怎样把内容图和风格图合成为一张新图，以及本项目为什么用它来批量制作训练图片。

神经风格迁移把一张图的内容和风格分开处理。Gatys 等人把内容定义为卷积网络高层特征的空间排列，把风格定义为多层特征之间的相关统计，再对每一对图片单独优化。每换一张内容图或风格图都要重新计算，因此不适合给 1349 张 Photo 训练图逐张制作增强样本。

后来的前馈方法把生成改成一次网络计算。实例归一化会减去每个通道自己的均值，再除以该通道的标准差，从而减弱单张图片原有的对比度。AdaIN 在此基础上，用风格图的均值和标准差替换内容图的均值和标准差。记内容特征为 fc，风格特征为 fs，每个通道在全部空间位置上的均值为 μ，标准差为 σ，则

AdaIN(fc, fs) = σ(fs) × (fc − μ(fc)) / σ(fc) + μ(fs)

替换之后，颜色和对比度来自风格图，空间排列仍来自内容图。测试时可以更换任意风格图，不必为每一种风格重新训练网络。

本项目的编码器是固定的 VGG-19，输出取到 relu4_1。解码器再把特征还原成图片。风格强度 α 取 0 到 1 之间的数：

t = α × AdaIN(fc, fs) + (1 − α) × fc

α 为 1 时，t 就是 AdaIN 的结果。本模块只使用已经训练好的编码器和解码器，不做新的风格迁移训练。准备好的四档强度是 0.25、0.5、0.75 和 1.0。主比较使用 1.0。

本节引用：Gatys 等，2016；Ulyanov 等，2016；Ulyanov 等，2017；Huang 和 Belongie，2017。

## 第二节　2.3 Style Augmentation for Image Classification

这一节说明风格化图片为什么可以当作分类训练数据。这里不展开 PACS 的文件划分，划分由数据集一节说明。

同一类别在照片、油画、卡通和素描里会有不同的纹理、笔触和对比度。卷积网络容易根据纹理做判断。训练时改变外观、同时保留原来的类别标签，可以让分类器见到更多风格下的同一类物体。

Jackson 等人在训练时随机采样风格向量，改变纹理、对比度和颜色，类别标签仍沿用内容图。Zhou 等人的 MixStyle 不生成新图片，而是在特征层混合两个训练样本的通道均值和方差。它使用的统计量和 AdaIN 相同，混合比例来自 Beta 分布，风格来自当前训练批次。本项目的 AdaIN 增强使用训练集以外的画作，并留下能够逐张查看的图片。

本项目的内容图只用 PACS 的 Photo 训练集。风格化图片保留原类别，再与原始照片一起训练分类器。要回答的问题是：这些新图片能否提高 Art、Cartoon 和 Sketch 上的准确率。普通 AdaIN 在整张图上使用同一个强度。主比较取 1.0，另外三档 0.25、0.5、0.75 用来观察强度变化。按图像结构改变每个位置强度的方法写在 3.5 节，两边使用同一批内容图和风格图。

本节引用：Geirhos 等，2019；Jackson 等，2019；Zhou 等，2021；Li 等，2017。

## 第三节　3.4 AdaIN Style Augmentation

这一节写普通 AdaIN 增强的具体流程、输出位置和目前已经生成的图片。

三种训练使用同一批内容图、同一个风格池和同一组配对。第一组只用原始 Photo。第二组加入本节的风格化图片。第三组加入结构保持后的风格化图片。第二组的主比较使用 α = 1。

内容清单是 `project_splits/photo_train.txt`，共 1349 张。清单里的标签是 1 到 7，读取时减 1，变成 0 到 6。风格图放在 `style_pool/`，共 10 张外部画作，取自 PyTorch AdaIN 的官方示例，来源与 AdaIN 论文使用的 WikiArt 画作一致。内容图仍是 PACS 的 Photo，不使用其他数据集的内容图。风格池去掉了接近素描测试域的 sketch.png，也不使用 PACS 里的 Art、Cartoon 和 Sketch。Photo 验证集、Photo 测试集和三个跨域测试集不参与风格化，也不用来挑选生成参数。

配对由种子 42 固定。10 张风格图先按这个种子打乱。第 i 张内容图使用打乱后第 i mod 10 张风格图。每张训练照片对应一张风格图，重复运行结果不变，10 张风格图都会被用到。

每一对图片按下面的顺序处理。

1. 转成 RGB，像素值放到 0 到 1。从文件读取时，把较短边缩放到 512。
2. 用冻结的 VGG-19 编码到 relu4_1，得到 fc 和 fs。权重文件是 `models/vgg_normalised.pth`。
3. 按通道计算均值和标准差，用第一节的 AdaIN 公式对齐。标准差计算中加入 0.00001，避免除以零。
4. 按第一节的 α 公式混合。主比较取 α = 1。四档强度 0.25、0.5、0.75、1.0 用于强度比较。
5. 用冻结的解码器还原成 RGB，并截断到 0 到 1。权重文件是 `models/decoder.pth`。

单张图片调用 `stylize(content, style, alpha)`。成员3如果要按位置调节强度，依次调用 `encode`、`mix`、`decode`。`mix` 可以接收一个数，也可以接收一张强度图。强度图会先缩放到特征图的高和宽，再按位置计算

t(x, y) = α(x, y) × AdaIN(fc, fs)(x, y) + (1 − α(x, y)) × fc(x, y)

3.5 节若使用 α(x, y) = α × (1 − β × M(x, y))，把这张图传入 `mix`。内容图、风格图、配对和输出尺寸保持不变。

只生成 α = 1 时，图片写到 `data/aug_adain/images/`，相对路径与原来的 Photo 相同。四档一起生成时，分别写到 `alpha_0.25`、`alpha_0.50`、`alpha_0.75`、`alpha_1.00`。清单标签仍是原来的 1 到 7。主比较的训练集是原始 1349 张加上 α = 1 的 1349 张，共 2698 张。结构保持组使用同样的 1349 对图片。

目前完成的是前 8 张训练照片的四档预览，共 32 张，总览图是 `data/aug_adain/samples/alpha_overview.jpg`。每一行从左到右是原图、风格图、0.25、0.50、0.75、1.00。这 32 张只用来检查效果，不放入训练集。1349 张全量图片尚未生成。

本节引用：Huang 和 Belongie，2017。权重来自 https://github.com/naoto0804/pytorch-AdaIN 。

## 第四节　4.2 AdaIN Augmentation Progress

这三条交给成员5，放进进度，不作为新的方法章节。

1. AdaIN 推理已经固定。编码器和解码器使用公开权重，训练时不更新。调用接口是 `stylize(content, style, alpha)`。`encode`、`mix`、`decode` 已经提供给成员3，空间强度图可以传入 `mix`。
2. 风格池固定为 10 张 PACS 以外的画作。种子 42 给 1349 张 Photo 训练图各配一张风格图。Art、Cartoon、Sketch，以及 Photo 的验证集和测试集，不参与风格化。四档强度是 0.25、0.5、0.75、1.0。主比较使用 1.0。
3. 流程图是 `figures/adain_flowchart.svg`。前 8 张内容图的四档预览在 `data/aug_adain/samples/`，共 32 张结果、对应对比图和一张总览图。全量 1349 张尚未生成，脚本是 `build_adain_aug.py`。

## 第五节　参考文献

阅读用的条目如下。BibTeX 不放在本文里，统一在 `refs.bib`。

- Gatys, L. A., Ecker, A. S., and Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks. CVPR, 2414–2423.
- Ulyanov, D., Vedaldi, A., and Lempitsky, V. (2016). Instance Normalization: The Missing Ingredient for Fast Stylization. arXiv:1607.08022.
- Ulyanov, D., Vedaldi, A., and Lempitsky, V. (2017). Improved Texture Networks: Maximizing Quality and Diversity in Feed-Forward Stylization and Texture Synthesis. CVPR, 6924–6932.
- Huang, X., and Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. ICCV, 1501–1510.
- Li, D., Yang, Y., Song, Y.-Z., and Hospedales, T. M. (2017). Deeper, Broader and Artier Domain Generalization. ICCV, 5542–5550.
- Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., and Brendel, W. (2019). ImageNet-trained CNNs are Biased towards Texture; Increasing Shape Bias Improves Accuracy and Robustness. ICLR.
- Jackson, P. T., Atapour-Abarghouei, A., Bonner, S., Breckon, T. P., and Obara, B. (2019). Style Augmentation: Data Augmentation via Style Randomization. CVPR Workshops, 83–92.
- Zhou, K., Yang, Y., Qiao, Y., and Xiang, T. (2021). Domain Generalization with MixStyle. ICLR.
