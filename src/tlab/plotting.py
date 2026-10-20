"""绘图工具（无 GUI 后端，直接存 PNG）。

阶段 0 只用到 loss 曲线；热图函数先放在这里，阶段 2 会用到。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前，否则 Windows 上会尝试开窗口
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_log(csv_path: str | Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with Path(csv_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if v == "" or v is None:
                    continue
                try:
                    cols.setdefault(k, []).append(float(v))
                except ValueError:
                    continue
    return cols


def plot_loss(csv_path: str | Path, out_path: str | Path, logy: bool = False) -> Path:
    d = read_log(csv_path)
    if "step" not in d or "train_loss" not in d:
        raise ValueError(f"{csv_path} 里没有 step/train_loss，先跑训练")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(d["step"], d["train_loss"], lw=1.0, alpha=0.75, label="train")

    xs = [s for s, v in zip(d["step"], d.get("val_loss", [])) if v is not None]
    ys = [v for v in d.get("val_loss", []) if v is not None]
    if xs and ys:
        ax.plot(xs, ys, "o-", ms=3, lw=1.4, label="val")

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats/char)")
    ax.set_title(f"loss — {Path(csv_path).parent.name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_attention(attn, tokens: list[str] | None = None, out_path: str | Path = "attn.png",
                   title: str = "attention", max_tokens: int = 48) -> Path:
    """attn: (T, T) 或 (H, T, T) 的 numpy/torch 张量。阶段 2 用。"""
    a = np.asarray(attn.detach().cpu() if hasattr(attn, "detach") else attn, dtype=float)
    if a.ndim == 2:
        a = a[None]
    a = a[:, :max_tokens, :max_tokens]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, a.shape[0], figsize=(3.2 * a.shape[0], 3.1), dpi=120, squeeze=False)
    for i, ax in enumerate(axes[0]):
        ax.imshow(a[i], cmap="viridis", aspect="auto")
        ax.set_title(f"head {i}", fontsize=8)
        ax.set_xlabel("key")
        ax.set_ylabel("query")
        if tokens is not None:
            ticks = range(0, min(len(tokens), max_tokens), 8)
            ax.set_xticks(list(ticks))
            ax.set_xticklabels([tokens[t] for t in ticks], fontsize=6)
            ax.set_yticks(list(ticks))
            ax.set_yticklabels([tokens[t] for t in ticks], fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
