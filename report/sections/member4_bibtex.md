Member 4 BibTeX entries (for the References section, member 5 to merge).
Member 1's five entries (zhou2023dgsurvey, zhou2021mixstyle, huang2017adain,
li2017dg, zhou2020dassl) are already in conference_101719.tex; the entries
below are the new ones my sections cite.

```bibtex
@inproceedings{he2016resnet,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016}
}

@inproceedings{zhang2018lpips,
  title     = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author    = {Zhang, Richard and Isola, Phillip and Efros, Alexei A. and Shechtman, Eli and Wang, Oliver},
  booktitle = {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018}
}

@article{canny1986,
  title   = {A Computational Approach to Edge Detection},
  author  = {Canny, John},
  journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
  volume  = {8},
  number  = {6},
  pages   = {679--698},
  year    = {1986}
}

@inproceedings{szegedy2016rethinking,
  title     = {Rethinking the Inception Architecture for Computer Vision},
  author    = {Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  booktitle = {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

@inproceedings{loshchilov2017sgdr,
  title     = {{SGDR}: Stochastic Gradient Descent with Warm Restarts},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {Proc. Int. Conf. Learning Representations (ICLR)},
  year      = {2017}
}
```

Notes:
- he2016resnet cited in 3.3 (ResNet-18).
- zhang2018lpips, canny1986 cited in 3.6 (structure metrics).
- szegedy2016rethinking (label smoothing), loshchilov2017sgdr (cosine schedule)
  cited in 3.3; both are training-protocol citations for reproducibility.

## Style-pool provenance (member 2 / member 3 to cite in 3.4 / 3.5)

The frozen style pool holds 10 external paintings from the public AdaIN demo
style set, which is drawn from the WikiArt collection used by the AdaIN paper.
Cite the AdaIN paper for the style domain; the concrete image files are listed
in `code/style_pool/SOURCES.txt` with their sha256, so the pool is
reproducible without a separate bibliography entry.

```bibtex
@misc{naoto0804pytorchadain,
  title        = {{pytorch-AdaIN}: {A} Reference Implementation of AdaIN},
  author       = {{Naoto0804}},
  howpublished = {\url{https://github.com/naoto0804/pytorch-AdaIN}},
  year         = {2026},
  note         = {Source of the 10 external style images in the frozen style pool}
}
```
