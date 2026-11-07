"""Rebuild scientific handoff figures from the implemented map and saved pilot.

From the repository root:
python -m member3_structure_preserving.handoff_member5.build_figures
Requires matplotlib in addition to the generator dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from member4_training_eval import config
from member3_structure_preserving.analysis import map_image
from member3_structure_preserving.structure_map import StructureConfig, build_structure_map, spatial_alpha


OUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE = "#275477", "#bd6c23"


def save(figure, name):
    figure.savefig(OUT / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    svg = OUT / f"{name}.svg"
    figure.savefig(svg, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    # Matplotlib paths include trailing spaces; keep the editable SVG clean in Git.
    svg.write_bytes(("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n").encode("utf-8"))
    plt.close(figure)


def maps():
    original = "photo/guitar/063_0024.jpg"
    with Image.open(config.IMAGE_ROOT / original) as source:
        content = TF.resize(source.convert("RGB"), 512, interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(content).unsqueeze(0)
    structure = build_structure_map(tensor)
    alpha = spatial_alpha(structure, 0.5, 0.7)
    content.save(OUT / "m3_content.png")
    map_image(structure).save(OUT / "m3_M.png")
    map_image(alpha).save(OUT / "m3_alpha.png")
    fig, axes = plt.subplots(1, 3, figsize=(5.1, 1.95), layout="constrained")
    axes[0].imshow(content)
    axes[0].set_title("(a) Content", fontsize=11)
    for ax, values, title in zip(axes[1:], (structure, alpha),
                                (r"(b) Structure $M$", r"(c) $\alpha(x,y)$")):
        artist = ax.imshow(values.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        bar = fig.colorbar(artist, ax=ax, fraction=0.047, pad=0.025, ticks=[0, 0.5, 1])
        bar.ax.tick_params(labelsize=10)
    for ax in axes:
        ax.set_axis_off()
    save(fig, "m3_structure_alpha")
    return {"original": original, "alpha0": 0.5, "beta": 0.7,
            "structure_config": StructureConfig().to_dict(),
            "content_size": list(content.size), "map_range": [0, 1],
            "alpha_range": [float(alpha.min()), float(alpha.max())]}


def alpha_rule():
    m = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5.7, 3.25), layout="constrained")
    palette = ["#8b949e", "#6193b4", "#37718f", ORANGE, "#173b52"]
    for beta, color in zip([0, 0.3, 0.5, 0.7, 1], palette):
        ax.plot(m, 0.5 * (1 - beta * m), label=f"{beta:g}", color=color,
                linewidth=2.6 if beta == 0.7 else 1.6)
    ax.set(xlim=(0, 1), ylim=(0, 0.55), xlabel=r"Structure response $M(x,y)$",
           ylabel=r"Local style strength $\alpha(x,y)$")
    ax.set_title(r"$\alpha(x,y)=\alpha_0[1-\beta M(x,y)]$,  $\alpha_0=0.5$", fontsize=12)
    ax.legend(title=r"$\beta$", frameon=False, fontsize=9, loc="lower left")
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "m3_alpha_rule")


def pipeline():
    fig, ax = plt.subplots(figsize=(11.8, 3.7))
    ax.set(xlim=(-0.025, 1.15), ylim=(0.02, 1.02))
    ax.axis("off")
    def box(x, y, text, width=0.15, height=0.15, color=BLUE):
        rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                              boxstyle="round,pad=0.007,rounding_size=0.012",
                              facecolor="#fff7ee" if color == ORANGE else "#f0f5f8",
                              edgecolor=color, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=12, color="#193042")
    def arrow(points, label=None, label_xy=None):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0],b[0]], [a[1],b[1]], color="#4d5c65", linewidth=1.1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=10, color="#4d5c65", linewidth=1.1))
        if label:
            ax.text(*label_xy, label, fontsize=11, ha="center", va="bottom", color="#263d4c")
    box(0.07, 0.66, r"Content $I_c$", width=0.16)
    box(0.07, 0.19, r"Style $I_s$", width=0.16)
    box(0.29, 0.66, "Frozen E", width=0.15)
    box(0.29, 0.19, "Frozen E", width=0.15)
    box(0.48, 0.42, "AdaIN", width=0.13)
    box(0.70, 0.42, "Feature mix", width=0.16, color=ORANGE)
    box(0.90, 0.42, "Frozen D", width=0.14)
    box(1.095, 0.42, r"$I_{sp}$", width=0.085)
    box(0.31, 0.91, r"Sobel $\rightarrow M$", width=0.21, height=0.15, color=ORANGE)
    box(0.66, 0.91, r"$\alpha_0(1-\beta M)$"+"\nArea downsample", width=0.25, height=0.17, color=ORANGE)
    arrow([(0.15,0.66),(0.215,0.66)])
    arrow([(0.15,0.19),(0.215,0.19)])
    arrow([(0.07,0.74),(0.07,0.91),(0.20,0.91)])
    arrow([(0.42,0.91),(0.527,0.91)])
    arrow([(0.365,0.66),(0.48,0.66),(0.48,0.503)], r"$F_c$", (0.415,0.68))
    arrow([(0.365,0.19),(0.48,0.19),(0.48,0.337)], r"$F_s$", (0.418,0.21))
    arrow([(0.40,0.66),(0.62,0.66),(0.66,0.503)])
    arrow([(0.72,0.818),(0.72,0.503)], r"$\alpha_f$", (0.762,0.72))
    arrow([(0.552,0.42),(0.612,0.42)])
    arrow([(0.788,0.42),(0.823,0.42)])
    arrow([(0.978,0.42),(1.045,0.42)])
    ax.text(0.73,0.11,r"$F_{mix}=\alpha_f\odot\mathrm{AdaIN}(F_c,F_s)+(1-\alpha_f)\odot F_c$",
            ha="center",fontsize=12,color="#193042")
    ax.text(0.29,0.04,"Shared encoder weights",ha="center",fontsize=10,color="#5b6870")
    save(fig, "m3_pipeline")


def beta_pilot():
    source = config.PROJECT_ROOT / "member3_structure_preserving/results/beta_sweep_7.json"
    runs = json.loads(source.read_text(encoding="utf-8"))["runs"]
    betas = [row["beta"] for row in runs]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), layout="constrained")
    for ax, key, color, ylabel in zip(axes, ["edge_score", "mean_alpha"], [BLUE, ORANGE],
                                    ["EdgeScore", "Mean local style strength"]):
        means = [r["ours"][key]["mean"] for r in runs]
        spreads = [r["ours"][key]["std"] for r in runs]
        ax.errorbar(betas,means,yerr=spreads,fmt="o-",color=color,capsize=3,linewidth=1.5)
        ax.set(xlabel=r"Protection $\beta$", ylabel=ylabel, xticks=betas)
        ax.grid(alpha=0.15)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylim(0.30,0.45)
    axes[1].set_ylim(0.25,0.55)
    fig.suptitle("Pilot only: 7 dog images; bars = image SD", fontsize=12)
    save(fig, "m3_beta_pilot")


def comparisons():
    sample = config.DATA_ROOT / "aug_sp/previews/samples_n7/a0p5_b0p7/generation.json"
    ledger = json.loads(sample.read_text(encoding="utf-8"))
    pool = json.loads(config.STYLE_POOL_MANIFEST.read_text(encoding="utf-8"))
    target = OUT / "examples"
    target.mkdir(exist_ok=True)
    index = []
    def rgb(path):
        with Image.open(path) as image:
            return image.convert("RGB")
    for original, entry in ledger["outputs"].items():
        style = pool["pairing"][original]
        style_path = config.STYLE_POOL_DIR / pool["pool"]["styles"][style]["file"]
        pictures = [rgb(config.IMAGE_ROOT / original), rgb(style_path)]
        for key in ("structure", "alpha"):
            with Image.open(config.DATA_ROOT / entry["files"][key]) as image:
                pictures.append(np.asarray(image.convert("L"), dtype=np.float32) / 255)
        pictures += [rgb(config.DATA_ROOT / entry["files"][key]) for key in ("adain", "ours")]
        fig, axes = plt.subplots(1,6,figsize=(11.8,2.45),layout="constrained")
        labels = ["(a) Content", "(b) Style", r"(c) $M$ [0,1]",
                  r"(d) $\alpha$ [0,1]", "(e) AdaIN", r"(f) SP, $\beta=0.7$"]
        for i, (ax, picture, title) in enumerate(zip(axes,pictures,labels)):
            if i in (2,3):
                ax.imshow(picture,cmap="gray",vmin=0,vmax=1)
            else:
                ax.imshow(picture)
            ax.set_title(title,fontsize=12)
            ax.set_axis_off()
        name = Path(original).parts[1] + "_comparison.png"
        fig.savefig(target/name,dpi=300,facecolor="white",bbox_inches="tight",pad_inches=0.08)
        plt.close(fig)
        index.append({"original":original,"style":style,"figure":"figures/examples/"+name,
                      "alpha0":0.5,"beta":0.7})
    shutil.copyfile(target/"guitar_comparison.png",OUT/"m3_guitar_comparison.png")
    return index


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "svg.fonttype":"none"})
    provenance = maps()
    alpha_rule()
    pipeline()
    beta_pilot()
    provenance["comparisons"] = comparisons()
    provenance["matplotlib_version"] = matplotlib.__version__
    (OUT / "figure_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
