# 成员3 → 成员5：一次性交付包

这是按成员3完整分工整理的报告与项目素材入口。压缩包为上一级的
`member3_handoff_member5.zip`，解压后先读本文件。四份英文LaTeX稿可直接移入总报告；
接口说明、公式和计划另用中文解释，便于成员5合并。代码、真实增强集和审计记录已完成；
本文没有把尚未进行的分类实验或LPIPS写成结果。

## 按分工找素材

| 要求 | 可直接使用的文件 |
|---|---|
| **2.4 Structure Preservation and Structure-Related Metrics 初稿** | [sections/02_4_related_work.tex](sections/02_4_related_work.tex) |
| **3.5 Structure-Preserving Style Augmentation 初稿** | [sections/03_5_method.tex](sections/03_5_method.tex) |
| **4.3 Current Status 两三条** | [sections/04_3_current_status.tex](sections/04_3_current_status.tex)，正好3条 |
| **5.3 Ablation and Analysis Plan 素材** | [sections/05_3_ablation_plan.tex](sections/05_3_ablation_plan.tex) 和 [BETA_PLAN.md](BETA_PLAN.md) |
| **M、自适应alpha和特征混合公式** | 本文下方公式、3.5初稿及 [INTERFACE.md](INTERFACE.md) |
| **独立M图和alpha图** | [figures/m3_M.png](figures/m3_M.png)、[figures/m3_alpha.png](figures/m3_alpha.png)、[对应内容图](figures/m3_content.png) |
| **七类真实对比样例** | [figures/examples/](figures/examples/)，每类一张、同内容/风格的AdaIN/SP比较 |
| **带标尺的M/alpha示意图** | [figures/m3_structure_alpha.png](figures/m3_structure_alpha.png)，另有SVG |
| **alpha规则曲线与模块流程图** | [figures/m3_alpha_rule.png](figures/m3_alpha_rule.png)、[figures/m3_pipeline.png](figures/m3_pipeline.png)，均有SVG |
| **β实验现状和计划** | [BETA_PLAN.md](BETA_PLAN.md)、[figures/m3_beta_pilot.png](figures/m3_beta_pilot.png) |
| **一次性模块接口说明** | [INTERFACE.md](INTERFACE.md) |
| **相关BibTeX** | [references.bib](references.bib)；当前手动参考文献可用 [references_manual.tex](references_manual.tex) |
| **排版预览** | [member3_handoff_preview.pdf](member3_handoff_preview.pdf) |

所有科学图来自实际代码或已保存实验记录，使用固定显示尺度。M和alpha的生成源为
Photo Training 的 `photo/guitar/063_0024.jpg`；不是手工绘制的虚构热力图。
`figures/figure_provenance.json` 记录来源和参数；`build_figures.py` 可复现这些图。

## 可以直接转交的4.3中文要点（3条）

1. **模块实现完成**：独立实现Sobel软结构图、自适应alpha、AdaIN特征混合、批量生成及beta扫描。
   固定复用成员2一次性交接的模型接口；12项测试通过，beta=0与普通AdaIN一致。
2. **正式数据就绪**：已生成1349张真实Photo训练图增强，位于 `data/aug_sp/`，
   alpha0=0.5、beta=0.7；解码、标签、冻结配对和哈希检查通过，给成员4的两类清单已准备。
   七类真实对比样例已提供。
3. **分析准备完成、分类待验证**：正式集EdgeScore为0.3436±0.0628（图片间SD，n=1349）；
   五组beta已在7张dog图上跑通，属于预实验。LPIPS和分类结果尚未完成，beta=0.7不是选优结果。

英文可粘贴版本在 `sections/04_3_current_status.tex`，不要把图片间SD当成分类器多seed标准差。

## 结构保持公式

令 `J = Gaussian_sigma1(gray(Ic))`，Sobel梯度幅值为

$$g=\sqrt{(K_x*J)^2+(K_y*J)^2}.$$

将不超过 `1e-6` 的幅值归零。用每张图的95%分位数归一化；分位数近零时改用最大幅值，
分母至少 `epsilon=1e-6`。得到截断响应 `g_hat` 后：

$$M=\mathrm{clip}_{[0,1]}\bigl(G_2*\mathrm{MaxPool}_{3\times3}(\hat g)\bigr),$$

$$\alpha(x,y)=\alpha_0[1-\beta M(x,y)],$$

$$\alpha_f=\mathrm{AreaResize}(\alpha,H_f,W_f),$$

$$F_{mix}=\alpha_f\odot\mathrm{AdaIN}(F_c,F_s)+(1-\alpha_f)\odot F_c,\qquad I_{sp}=D(F_{mix}).$$

M是边缘响应，不是语义显著性或前景分割。beta=0退化成普通AdaIN；alpha0=0为网络重建。
beta增大还会降低平均alpha，因此必须区分空间保护和整体风格减弱的作用。

## 接收一次接口，之后独立完成

成员2接口固定到 `416bdc9811d67aae80fd79e8fac1ae0f37d5f339` 的 `adain.py`：
RGB `[0,1]`、`prepare_image/encode/mix/decode` 和 `stylize(content,style,alpha)`。
成员3在本地完成地图、参数规则、实验、图像与清单生成，不依赖成员2的个人目录或批量脚本。
风格池和配对直接读取成员4已冻结的公共文件。完整约定见 `INTERFACE.md`。

## 给成员5的合并步骤

1. 将四份 `sections/*.tex` 分别放入2.4、3.5、4.3和5.3的逻辑位置。
   文件用 `\subsection`，不自行强行设置章节计数；IEEE格式会显示为II-D、III-E等。
   5.3是成员3贡献，应与成员4训练计划合并，避免重复创建同名小节。
2. 将 `figures/` 中使用的图复制到报告图目录，或在总报告添加：

   ```latex
   \graphicspath{{figures/}{../member3_structure_preserving/handoff_member5/figures/}}
   ```

   路径假定在 `report/` 目录编译。3.5中的流程图用跨双栏的 `figure*`；M/alpha图用单栏。
3. 当前总报告用手动 `thebibliography`，已有 `huang2017adain`、`zhang2018lpips`、`canny1986`。
   可复用这三个键，只把 `references_manual.tex` 中新增的两条合入现有环境。
   若统一改用BibTeX，则合并 `references.bib` 并按键去重，不同时使用两套参考文献管理。
4. 4.3只报告已完成事项。5.3保留将来时：完整beta消融、LPIPS、均值强度匹配控制和分类实验仍待做。
   不把7张dog预实验写成全类别结论，不根据目标域测试分数选alpha/beta。
5. 校对现有3.2中Photo Validation“选择checkpoint”的旧描述，使其与成员4的固定预算、
   final-checkpoint协议一致。增强组分类实验仍待成员4执行，不能改成已得到提升。

`report/sections/member3_method.tex` 已改为本包3.5稿的兼容入口，二者选择一个插入，不要重复。
没有自动改写总报告其他成员段落，最终章节顺序和去重由成员5完成。

预览PDF采用通用双栏字体和阿拉伯编号，便于直接阅读；正文片段合并后使用总报告的IEEE排版。
参考文献已通过BibTeX编译；预览不替代成员5对整篇报告的最终编译。
现成稿件和图可直接使用。图像重建需要完整仓库、PACS及已生成的七类样例；
运行 `python -m member3_structure_preserving.handoff_member5.build_figures`，需安装matplotlib。
`build_preview.py` 可重新编译PDF，需本机LaTeX、BibTeX和pdftoppm。

## 本包自带的证据

`evidence/` 包含全量验收记录、生成参数和哈希、全量结构指标以及beta预实验汇总。
大体积原图、模型和1349张增强图不在报告素材包内。
`evidence/generation.json` 的输入/输出路径相对于原实验DATA_ROOT，不意味着包内包含这些图片。
`MANIFEST.sha256` 列出本包各文件哈希；原始数据及代码运行证据与报告素材的哈希含义不同，
以各文件注释为准。

## 引文核对来源

仅使用原论文和官方文档；2.4明确承认AdaIN原论文本身已有空间控制，
本项目贡献是边缘图驱动的具体轻量实现与受控比较，不宣称首次提出空间风格控制。

- [Gatys等，CVPR 2016原文](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- [Huang与Belongie，AdaIN原文](https://arxiv.org/abs/1703.06868)
- [Zhang等，LPIPS原文](https://arxiv.org/abs/1801.03924)
- [Canny，IEEE原论文](https://doi.org/10.1109/TPAMI.1986.4767851)
- [OpenCV官方Sobel说明](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)

这些引用支持方法背景和指标定义；1349张、EdgeScore、beta预实验及验收结果来自本项目审计文件，
不是文献中的数字。完整增强图和模型权重按仓库约定留在本地，不包含在报告素材压缩包里。
