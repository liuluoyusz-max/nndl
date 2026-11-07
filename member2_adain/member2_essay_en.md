# Member 2 Report Draft (English)

Read this file. The typeset source is `member2_essay_en.tex`. The reference database remains `refs.bib`. The section numbers match the Chinese draft.

| Report section | This file | Typeset source |
|---|---|---|
| 2.2 Style Transfer and AdaIN | Section 1 | `member2_essay_en.tex`, section 2.2 |
| 2.3 Style Augmentation for Image Classification | Section 2 | `member2_essay_en.tex`, section 2.3 |
| 3.4 AdaIN Style Augmentation | Section 3 | `member2_essay_en.tex`, section 3.4 |
| 4.2 AdaIN Augmentation Progress | Section 4 | `member2_essay_en.tex`, section 4.2 |
| References | Section 5 | `refs.bib` |

## Scope

The implementation covers plain AdaIN only.

- The call is `stylize(content, style, alpha)`. Member 3 also uses `encode`, `mix`, and `decode`.
- The style pool has 10 external paintings. The random seed is 42. Only the 1,349 Photo training images are processed.
- The four strengths are 0.25, 0.5, 0.75, and 1.0. The main three-way comparison uses 1.0.
- Previews for 8 content images are done, giving 32 stylized images. The full set of 1,349 images has not been generated.

The report text covers Sections 2.2, 2.3, and 3.4, plus three progress items for Section 4.2. The structure map, the parameter beta, classifier training, and the accuracy table are written by other members.

## Section 1. 2.2 Style Transfer and AdaIN

This section explains how AdaIN combines a content image and a style image, and why the project uses it to build training images in batches.

Neural style transfer separates the content of an image from its style. Gatys et al. define content as the spatial arrangement of higher-layer convolutional features, and style as correlation statistics across feature maps. Their method optimizes each content-style pair separately. A new content image or a new style image requires a new optimization, so the method is not practical for making one augmented sample for each of the 1,349 Photo training images.

Later feed-forward methods replace that optimization with one network pass. Instance normalization subtracts each channel's own mean and divides by that channel's standard deviation, which reduces the contrast particular to the content image. AdaIN then replaces the content mean and standard deviation with those of the style image. Let fc be the content features and fs the style features. For each channel, μ and σ are the mean and standard deviation over all spatial locations:

AdaIN(fc, fs) = σ(fs) × (fc − μ(fc)) / σ(fc) + μ(fs)

After this replacement, color and contrast come from the style image, while the spatial layout still comes from the content image. Any style image can be used at test time, without training a separate network for each style.

In this project the encoder is a fixed VGG-19, and its output is taken at relu4_1. A decoder turns the features back into an image. The style strength α is a number between 0 and 1:

t = α × AdaIN(fc, fs) + (1 − α) × fc

When α is 1, t is the AdaIN output. This module uses pretrained encoder and decoder weights and does not train a new style-transfer network. The four prepared strengths are 0.25, 0.5, 0.75, and 1.0. The main comparison uses 1.0.

Citations for this section: Gatys et al., 2016; Ulyanov et al., 2016; Ulyanov et al., 2017; Huang and Belongie, 2017.

## Section 2. 2.3 Style Augmentation for Image Classification

This section explains why stylized images can be used as training data for a classifier. The PACS file split is described in the dataset section, not here.

The same object category can appear with different texture, brush strokes, and contrast in photos, paintings, cartoons, and sketches. Convolutional networks often rely on texture. Changing appearance during training, while keeping the original class label, lets the classifier see the same category under more styles.

Jackson et al. sample a random style vector during training. The sampled style changes texture, contrast, and color, while the class label remains that of the content image. Zhou et al. do not synthesize a new image in MixStyle. They mix the channel mean and variance of two training samples in feature space. Those statistics are the same ones used by AdaIN. The mixing weight comes from a Beta distribution, and the style comes from the current training batch. The AdaIN augmentation in this project uses paintings from outside the training set and keeps an image that can be inspected one by one.

Content images come only from the PACS Photo training set. Each stylized image keeps the original class and is then trained together with the original photo. The question is whether these new images raise accuracy on Art, Cartoon, and Sketch. Plain AdaIN uses one strength for the whole image. The main comparison uses 1.0. The other three values, 0.25, 0.5, and 0.75, are used to observe the effect of strength. The method that changes strength at each location according to image structure is written in Section 3.5. Both methods use the same content images and the same style images.

Citations for this section: Geirhos et al., 2019; Jackson et al., 2019; Zhou et al., 2021; Li et al., 2017.

## Section 3. 3.4 AdaIN Style Augmentation

This section states the procedure for plain AdaIN augmentation, where the outputs are stored, and which images have already been generated.

The three training settings share the same content images, the same style pool, and the same content-style pairs. The first setting uses the original Photo images only. The second adds the stylized images from this section. The third adds the structure-preserving stylized images. The main comparison for the second setting uses α = 1.

The content list is `project_splits/photo_train.txt`, with 1,349 images. Labels in the list run from 1 to 7. The data loader subtracts 1, so the labels used by the model run from 0 to 6. Style images are the 10 external paintings in `style_pool/`, taken from the official PyTorch AdaIN examples. Their source is the same WikiArt line used by the AdaIN paper. Content images remain PACS Photo images. The pool excludes `sketch.png`, which is close to the Sketch test domain, and it does not use PACS Art, Cartoon, or Sketch images. The Photo validation set, the Photo test set, and the three cross-domain test sets are not stylized and are not used to choose generation settings.

Pairs are fixed by seed 42. The 10 style images are shuffled with that seed. Content image i receives the style at position i mod 10 in the shuffled list. Each training photo is paired with one style image. Repeating the run gives the same pairs, and all 10 styles are used.

Each pair is processed as follows.

1. Convert the image to RGB and scale pixel values to the range 0 to 1. When the image is read from a file, resize its shorter side to 512.
2. Encode both images to relu4_1 with a frozen VGG-19, producing fc and fs. The weight file is `models/vgg_normalised.pth`.
3. Compute the per-channel mean and standard deviation, and align them with the AdaIN formula in Section 1. The value 0.00001 is added inside the standard deviation so that the denominator is not zero.
4. Blend the features with the alpha formula in Section 1. The main comparison uses α = 1. The four strengths 0.25, 0.5, 0.75, and 1.0 are used for the strength comparison.
5. Decode the features to RGB with a frozen decoder and clip the result to the range 0 to 1. The weight file is `models/decoder.pth`.

A single image is produced by `stylize(content, style, alpha)`. To vary strength by location, Member 3 calls `encode`, `mix`, and `decode` in that order. `mix` accepts either one number or a strength map. The map is resized to the height and width of the feature map, and the blend at each location is

t(x, y) = α(x, y) × AdaIN(fc, fs)(x, y) + (1 − α(x, y)) × fc(x, y)

If Section 3.5 uses α(x, y) = α × (1 − β × M(x, y)), that map is passed to `mix`. The content images, style images, pairs, and output size stay the same.

When only α = 1 is generated, images are written to `data/aug_adain/images/` with the same relative paths as the original Photo images. When all four strengths are generated, they are written separately to `alpha_0.25`, `alpha_0.50`, `alpha_0.75`, and `alpha_1.00`. List labels remain the original values from 1 to 7. The main comparison trains on the original 1,349 images plus the 1,349 images at α = 1, which is 2,698 images in total. The structure-preserving setting uses the same 1,349 pairs.

What is finished now is a preview of the first 8 training photos at the four strengths, 32 images in total. The overview is `data/aug_adain/samples/alpha_overview.jpg`. Each row shows, from left to right, the original image, the style image, and the results at 0.25, 0.50, 0.75, and 1.00. These 32 images are for visual checking only. They are not added to the training set. The full set of 1,349 images has not been generated.

Citation for this section: Huang and Belongie, 2017. The weights come from https://github.com/naoto0804/pytorch-AdaIN .

## Section 4. 4.2 AdaIN Augmentation Progress

These three items go to Member 5 for the progress section. They are not a new method section.

1. AdaIN inference is fixed. The encoder and decoder use public weights and are not updated during classifier training. The call is `stylize(content, style, alpha)`. `encode`, `mix`, and `decode` are available to Member 3, and a spatial strength map can be passed to `mix`.
2. The style pool is fixed at 10 paintings outside PACS. Seed 42 assigns one style image to each of the 1,349 Photo training images. Art, Cartoon, Sketch, and the Photo validation and test sets are not stylized. The four strengths are 0.25, 0.5, 0.75, and 1.0. The main comparison uses 1.0.
3. The flowchart is `figures/adain_flowchart.svg`. Four-strength previews of the first 8 content images are in `data/aug_adain/samples/`: 32 results, the matching side-by-side comparisons, and one overview. The full set of 1,349 images has not been generated. The script is `build_adain_aug.py`.

## Section 5. References

The readable entries are listed below. The BibTeX database is `refs.bib`.

- Gatys, L. A., Ecker, A. S., and Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks. CVPR, 2414–2423.
- Ulyanov, D., Vedaldi, A., and Lempitsky, V. (2016). Instance Normalization: The Missing Ingredient for Fast Stylization. arXiv:1607.08022.
- Ulyanov, D., Vedaldi, A., and Lempitsky, V. (2017). Improved Texture Networks: Maximizing Quality and Diversity in Feed-Forward Stylization and Texture Synthesis. CVPR, 6924–6932.
- Huang, X., and Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. ICCV, 1501–1510.
- Li, D., Yang, Y., Song, Y.-Z., and Hospedales, T. M. (2017). Deeper, Broader and Artier Domain Generalization. ICCV, 5542–5550.
- Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., and Brendel, W. (2019). ImageNet-trained CNNs are Biased towards Texture; Increasing Shape Bias Improves Accuracy and Robustness. ICLR.
- Jackson, P. T., Atapour-Abarghouei, A., Bonner, S., Breckon, T. P., and Obara, B. (2019). Style Augmentation: Data Augmentation via Style Randomization. CVPR Workshops, 83–92.
- Zhou, K., Yang, Y., Qiao, Y., and Xiang, T. (2021). Domain Generalization with MixStyle. ICLR.
