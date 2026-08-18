"""统一论文图表样式并处理中文与数学符号。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager


FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
ENGLISH_FONT_CANDIDATES = [
    "Times New Roman",
    "Liberation Serif",
    "Nimbus Roman",
    "DejaVu Serif",
]


def configure_plot_style() -> str:
    """选择当前系统可用的中文字体，返回实际字体名。"""

    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in FONT_CANDIDATES if font in available), "DejaVu Sans")
    matplotlib.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    return selected


def configure_research_plot_style(language: str = "zh") -> str:
    """配置适合论文独立图的中英文科研样式，并返回实际使用的字体。"""

    if language not in {"zh", "en"}:
        raise ValueError("科研绘图语言只支持zh或en。")
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = FONT_CANDIDATES if language == "zh" else ENGLISH_FONT_CANDIDATES
    fallback = "DejaVu Sans" if language == "zh" else "DejaVu Serif"
    selected = next((font for font in candidates if font in available), fallback)
    family = "sans-serif" if language == "zh" else "serif"
    matplotlib.rcParams.update(
        {
            "font.family": family,
            f"font.{family}": [selected, fallback],
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 9.0,
            "lines.linewidth": 1.8,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    return selected


def create_encoding_smoke_plot(directory: str | Path) -> dict[str, str]:
    """生成包含中文、负数和 MathText 公式的 PNG/PDF。"""

    selected_font = configure_plot_style()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot([0, 1, 2, 3], [-0.05, 0.3, 0.7, 1.25], marker="o")
    axis.set_title("专家—AI 共识编码测试")
    axis.set_xlabel(r"理论调整量 $\delta^*_{k,t}$")
    axis.set_ylabel(r"共识度 $ACD_k$")
    axis.grid(alpha=0.25)
    figure.tight_layout()

    png = root / "encoding_plot.png"
    pdf = root / "encoding_plot.pdf"
    figure.savefig(png, dpi=160)
    figure.savefig(pdf)
    plt.close(figure)

    return {"font": selected_font, "png": str(png), "pdf": str(pdf)}
