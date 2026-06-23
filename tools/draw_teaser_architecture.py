from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"
TEASER_BASE = ROOT / "private" / "teaser_assets" / "teaser_final_trials" / "blank_center_1p6.png"
TEASER_TITLE = "Hand Geometry Enhanced Fusion"
TEASER_CAPTIONS = [
    ("Fisheye View", 320),
    ("Hand Geometry", 1984),
]


COLORS = {
    "ink": "#22313f",
    "muted": "#667085",
    "line": "#98a2b3",
    "bg": "#fbfcfe",
    "panel": "#ffffff",
    "anchor": "#2f80ed",
    "anchor_light": "#eef6ff",
    "geometry": "#f2994a",
    "geometry_light": "#fff4e8",
    "residual": "#667085",
    "residual_light": "#f2f4f7",
    "output": "#219653",
    "output_light": "#ecf8f0",
}

BOX_SCALE = 0.75
GROUP_BOX_SCALE = 0.84


def box(ax, xy, width, height, label, *, fc, ec, lw=1.2, fontsize=8.8, weight="medium"):
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2
    width *= BOX_SCALE
    height *= BOX_SCALE
    xy = (cx - width / 2, cy - height / 2)
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["ink"],
        linespacing=1.15,
    )
    return patch


def label(ax, x, y, text, *, size=8.0, color=None, weight="regular", ha="center", shield=False):
    bbox = None
    if shield:
        bbox = {
            "boxstyle": "round,pad=0.08,rounding_size=0.02",
            "facecolor": COLORS["bg"],
            "edgecolor": "none",
            "alpha": 0.92,
        }
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        color=color or COLORS["muted"],
        fontweight=weight,
        bbox=bbox,
        zorder=8 if shield else None,
    )


def group_box(ax, xy, width, height, title, *, fc, ec, title_color=None):
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2
    width *= GROUP_BOX_SCALE
    height *= GROUP_BOX_SCALE
    xy = (cx - width / 2, cy - height / 2)
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height - 0.024,
        title,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="semibold",
        color=title_color or COLORS["ink"],
    )
    return patch


def arrow(ax, start, end, *, color=None, lw=1.25, rad=0.0, alpha=1.0, ms=9):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color or COLORS["line"],
        alpha=alpha,
        shrinkA=4,
        shrinkB=5,
    )
    ax.add_patch(patch)
    return patch


def token(ax, x, y, text, *, kind="anchor", width=0.095, height=0.052, size=7.6):
    if kind == "anchor":
        fc, ec = COLORS["anchor_light"], COLORS["anchor"]
    elif kind == "geometry":
        fc, ec = COLORS["geometry_light"], COLORS["geometry"]
    elif kind == "output":
        fc, ec = COLORS["output_light"], COLORS["output"]
    else:
        fc, ec = COLORS["residual_light"], COLORS["residual"]
    return box(ax, (x - width / 2, y - height / 2), width, height, text, fc=fc, ec=ec, fontsize=size, lw=1.0)


def scaled_half(width: float, scale: float = BOX_SCALE) -> float:
    return width * scale / 2


def scaled_bbox(xy, width: float, height: float, scale: float):
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2
    actual_w = width * scale
    actual_h = height * scale
    left = cx - actual_w / 2
    right = cx + actual_w / 2
    bottom = cy - actual_h / 2
    top = cy + actual_h / 2
    return left, right, bottom, top


def even_centers(left: float, right: float, count: int, margin: float):
    if count == 1:
        return [(left + right) / 2]
    usable_left = left + margin
    usable_right = right - margin
    step = (usable_right - usable_left) / (count - 1)
    return [usable_left + i * step for i in range(count)]


def draw_architecture(output_prefix: Path, *, figsize=(10.2, 4.6), dpi=300) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.0,
        }
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    # Main stages. This figure is intended to sit inside a three-panel teaser,
    # so it deliberately avoids a standalone title while keeping paper-figure
    # information density.
    box(
        ax,
        (0.04, 0.62),
        0.155,
        0.18,
        "Raw AR\nSegment\nvideo + sensors",
        fc=COLORS["panel"],
        ec="#d0d5dd",
        fontsize=8.3,
    )
    label(ax, 0.118, 0.555, "time-aligned clips", size=6.4, shield=True)
    box(
        ax,
        (0.04, 0.25),
        0.155,
        0.20,
        "Feature\nExtraction\nper modality",
        fc=COLORS["panel"],
        ec="#d0d5dd",
        fontsize=8.3,
    )
    arrow(ax, (0.12, 0.60), (0.12, 0.46), color=COLORS["line"], lw=1.15)

    # Feature rows.
    rows = [
        ("Gesture\nvideo", "MediaPipe\nlandmarks", "Hand Geometry\n10 x 96", "geometry", 0.80),
        ("Scene\nframe", "ViT", "Scene\nfeature", "anchor", 0.67),
        ("ASR\ntext", "Whisper", "Sentence\nembedding", "anchor", 0.54),
        ("Audio", "MFCC", "Audio\nfeature", "residual", 0.36),
        ("IMU", "10-frame", "Temporal\nsensor", "residual", 0.24),
    ]
    for left, middle, right, kind, y in rows:
        left_w = 0.088
        middle_w = 0.092 if "\n" in middle else 0.066
        right_w = 0.10
        token(ax, 0.285, y, left, kind=kind, width=left_w, height=0.056 if "\n" in left else 0.045, size=6.55)
        token(ax, 0.39, y, middle, kind=kind, width=middle_w, height=0.056 if "\n" in middle else 0.045, size=6.35)
        token(ax, 0.505, y, right, kind=kind, width=right_w, height=0.056 if "\n" in right else 0.045, size=6.35)
        source_y = 0.315 if y < 0.46 else 0.71
        left_edge = 0.285 - scaled_half(left_w)
        left_right = 0.285 + scaled_half(left_w)
        middle_left = 0.39 - scaled_half(middle_w)
        middle_right = 0.39 + scaled_half(middle_w)
        right_left = 0.505 - scaled_half(right_w)
        arrow(
            ax,
            (0.178, source_y),
            (left_edge - 0.006, y),
            color=COLORS["line"],
            rad=0.12 if y > 0.56 else -0.12,
            lw=1.05,
        )
        arrow(ax, (left_right + 0.004, y), (middle_left - 0.004, y), color=COLORS["line"], lw=1.0, ms=7)
        arrow(ax, (middle_right + 0.004, y), (right_left - 0.004, y), color=COLORS["line"], lw=1.0, ms=7)

    # Token groups.
    anchor_group_xy = (0.615, 0.665)
    anchor_group_w = 0.21
    anchor_group_h = 0.17
    group_box(ax, anchor_group_xy, anchor_group_w, anchor_group_h, "Anchor Tokens", fc="#f8fbff", ec="#b7d9ff")
    anchor_left, anchor_right, anchor_bottom, anchor_top = scaled_bbox(
        anchor_group_xy, anchor_group_w, anchor_group_h, GROUP_BOX_SCALE
    )
    anchor_xs = even_centers(anchor_left, anchor_right, 3, margin=0.024)
    anchor_token_y = anchor_bottom + 0.055
    token(ax, anchor_xs[0], anchor_token_y, "Gesture", kind="geometry", width=0.041, height=0.04, size=6.35)
    token(ax, anchor_xs[1], anchor_token_y, "Text", kind="anchor", width=0.031, height=0.04, size=6.35)
    token(ax, anchor_xs[2], anchor_token_y, "Scene", kind="anchor", width=0.035, height=0.04, size=6.35)
    label(ax, (anchor_left + anchor_right) / 2, anchor_bottom + 0.007, "gesture / text / scene", size=5.8, shield=True)

    residual_group_xy = (0.615, 0.175)
    residual_group_w = 0.21
    residual_group_h = 0.15
    group_box(ax, residual_group_xy, residual_group_w, residual_group_h, "Residual Support", fc="#fbfbfc", ec="#d0d5dd")
    residual_left, residual_right, residual_bottom, residual_top = scaled_bbox(
        residual_group_xy, residual_group_w, residual_group_h, GROUP_BOX_SCALE
    )
    residual_xs = even_centers(residual_left, residual_right, 2, margin=0.046)
    residual_token_y = residual_bottom + 0.049
    token(ax, residual_xs[0], residual_token_y, "IMU", kind="residual", width=0.052, height=0.04, size=6.55)
    token(ax, residual_xs[1], residual_token_y, "Audio", kind="residual", width=0.06, height=0.04, size=6.55)
    label(ax, (residual_left + residual_right) / 2, residual_bottom + 0.006, "gated support", size=5.8, shield=True)

    # Fusion core.
    box(ax, (0.62, 0.415), 0.20, 0.145, "Perceiver-style\nLatent Fusion", fc="#ffffff", ec=COLORS["anchor"], fontsize=8.45, lw=1.35)
    dot_xs = [0.676, 0.699, 0.722, 0.745, 0.768]
    for i, x in enumerate(dot_xs):
        dot = Circle((x, 0.438), radius=0.0073, facecolor=COLORS["anchor"] if i != 0 else COLORS["geometry"], edgecolor="none")
        ax.add_patch(dot)
    label(ax, 0.72, 0.372, "latent tokens", size=5.9, shield=True)

    # Connections into fusion.
    fusion_left, fusion_right, fusion_bottom, fusion_top = scaled_bbox((0.62, 0.415), 0.20, 0.145, BOX_SCALE)
    feature_right = 0.505 + scaled_half(0.10)
    arrow_gap = 0.014
    arrow(ax, (feature_right + arrow_gap, 0.80), (fusion_left - arrow_gap, 0.526), color=COLORS["geometry"], lw=1.45, rad=-0.15, alpha=0.95)
    arrow(ax, (feature_right + arrow_gap, 0.67), (fusion_left - arrow_gap, 0.505), color=COLORS["anchor"], lw=1.25, rad=-0.08, alpha=0.95)
    arrow(ax, (feature_right + arrow_gap, 0.54), (fusion_left - arrow_gap, 0.486), color=COLORS["anchor"], lw=1.25, rad=-0.02, alpha=0.95)
    arrow(ax, (feature_right + arrow_gap, 0.36), (fusion_left - arrow_gap, 0.462), color=COLORS["residual"], lw=1.2, rad=0.07, alpha=0.9)
    arrow(ax, (feature_right + arrow_gap, 0.24), (fusion_left - arrow_gap, 0.443), color=COLORS["residual"], lw=1.2, rad=0.12, alpha=0.9)
    arrow(ax, ((anchor_left + anchor_right) / 2, anchor_bottom - 0.01), ((anchor_left + anchor_right) / 2, fusion_top + 0.014), color=COLORS["anchor"], lw=1.3)
    arrow(ax, ((residual_left + residual_right) / 2, residual_top + 0.01), ((residual_left + residual_right) / 2, fusion_bottom - 0.014), color=COLORS["residual"], lw=1.2)

    # Outputs.
    box(ax, (0.875, 0.60), 0.092, 0.09, "Intent\nHead", fc=COLORS["output_light"], ec=COLORS["output"], fontsize=7.5)
    box(ax, (0.875, 0.455), 0.092, 0.09, "Scene\nHead", fc=COLORS["output_light"], ec=COLORS["output"], fontsize=7.5)
    box(ax, (0.855, 0.255), 0.13, 0.12, "Final Joint\nPrediction\n12 classes", fc="#ffffff", ec=COLORS["output"], fontsize=7.5, lw=1.25)
    label(ax, 0.922, 0.713, "6 intents", size=6.3, color=COLORS["output"])
    label(ax, 0.922, 0.565, "2 scenes", size=6.3, color=COLORS["output"])
    fusion_right = 0.795
    intent_left = 0.887
    scene_left = 0.887
    joint_top = 0.36
    arrow(ax, (fusion_right + 0.008, 0.518), (intent_left - 0.008, 0.645), color=COLORS["output"], rad=0.08, lw=1.35)
    arrow(ax, (fusion_right + 0.008, 0.488), (scene_left - 0.008, 0.50), color=COLORS["output"], lw=1.35)
    arrow(ax, (0.982, 0.635), (0.975, joint_top + 0.006), color=COLORS["output"], rad=-0.28, lw=1.35)
    arrow(ax, (0.922, 0.467), (0.922, joint_top + 0.004), color=COLORS["output"], rad=0.0, lw=1.35)

    # Centered legend. For future tweaks, prefer smaller boxes over smaller
    # text so the teaser remains readable after being pasted into the mockup.
    legend_y = 0.105
    legend = [
        ("anchor", COLORS["anchor"]),
        ("hand geometry", COLORS["geometry"]),
        ("residual", COLORS["residual"]),
        ("prediction", COLORS["output"]),
    ]
    x_positions = [0.28, 0.44, 0.615, 0.805]
    for (name, color), x in zip(legend, x_positions):
        ax.add_patch(Circle((x, legend_y), radius=0.0068, facecolor=color, edgecolor="none"))
        label(ax, x + 0.012, legend_y, name, size=6.0, ha="left")

    # Tighten the internal composition for the teaser center panel.
    ax.set_xlim(0.02, 1.025)
    ax.set_ylim(0.075, 0.875)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, format="pdf", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, format="png", dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png_path


def resize_contain(image, target_size, bg=(251, 252, 254)):
    from PIL import Image

    target_w, target_h = target_size
    scale = min(target_w / image.width, target_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, bg)
    offset = ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2)
    canvas.paste(resized.convert("RGB"), offset)
    return canvas


def make_full_mockup(arch_png: Path) -> Path | None:
    if not TEASER_BASE.exists():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    base = Image.open(TEASER_BASE).convert("RGB")
    arch = Image.open(arch_png).convert("RGB")
    center_w = base.width - 2 * base.height * 4 // 3
    if center_w <= 0:
        center_w = base.width - 1280
    left_w = (base.width - center_w) // 2
    center = resize_contain(arch, (center_w, base.height), bg=(251, 252, 254))
    output = base.copy()
    output.paste(center, (left_w, 0))
    out_path = FIGURE_DIR / "teaser_full_mockup.png"
    output.save(out_path, quality=96)
    return out_path


def make_titled_mockup(mockup_path: Path | None) -> Path | None:
    if mockup_path is None:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    base = Image.open(mockup_path).convert("RGB")
    title_h = 54
    output = Image.new("RGB", (base.width, base.height + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(output)

    font = None
    for font_path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, 22)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    caption_font = font
    for font_path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]:
        try:
            caption_font = ImageFont.truetype(font_path, 19)
            break
        except OSError:
            continue

    baseline_y = title_h - 15
    title_bbox = draw.textbbox((0, 0), TEASER_TITLE, font=font)
    title_w = title_bbox[2] - title_bbox[0]
    title_h_text = title_bbox[3] - title_bbox[1]
    draw.text(((base.width - title_w) // 2, baseline_y - title_h_text), TEASER_TITLE, fill=COLORS["ink"], font=font)

    for caption, center_x in TEASER_CAPTIONS:
        bbox = draw.textbbox((0, 0), caption, font=caption_font)
        caption_w = bbox[2] - bbox[0]
        caption_h = bbox[3] - bbox[1]
        draw.text((center_x - caption_w // 2, baseline_y - caption_h), caption, fill=COLORS["muted"], font=caption_font)

    output.paste(base, (0, title_h))

    out_path = FIGURE_DIR / "teaser_full_mockup_with_title.png"
    output.save(out_path, quality=96)
    return out_path


def main() -> None:
    arch_png = draw_architecture(FIGURE_DIR / "teaser_architecture")
    mockup = make_full_mockup(arch_png)
    titled_mockup = make_titled_mockup(mockup)
    print(f"[saved] {FIGURE_DIR / 'teaser_architecture.svg'}")
    print(f"[saved] {FIGURE_DIR / 'teaser_architecture.pdf'}")
    print(f"[saved] {arch_png}")
    if mockup is not None:
        print(f"[saved] {mockup}")
    if titled_mockup is not None:
        print(f"[saved] {titled_mockup}")
    else:
        print("[skip] teaser_full_mockup.png or title variant (base image or Pillow not available)")


if __name__ == "__main__":
    main()
