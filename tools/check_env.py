"""环境门禁：一条命令确认"这个项目在这台机器上真的能跑"。

检查的不是版本号好看，而是实际能力：
  - numpy<->tensor 双向桥（numpy 2 与 torch 2.2 的 ABI 冲突就在这里）
  - matplotlib 能不能把 tensor 数据画成图（阶段 1/2 的产出全靠它）
  - CUDA 能不能真的算，bf16 混合精度能不能用
  - 注意力后端哪些可用（阶段 3 的对比口径要写清楚）
  - pytest 在不在

用法:  .\run.ps1 -m tlab env     或     python tools/check_env.py
"""

from __future__ import annotations

import os
import platform
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "runs" / "_env"
RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            RESULTS.append((name, True, fn()))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn

    return deco


@check("python")
def _python():
    assert sys.version_info[:2] >= (3, 12), f"需要 3.12+，当前 {sys.version.split()[0]}"
    return f"{platform.python_version()} @ {sys.executable}"


@check("numpy_abi")
def _numpy_abi():
    import numpy as np
    import torch

    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    n = t.numpy()
    assert n.sum() == 15, n
    back = torch.from_numpy(np.arange(4, dtype="float32"))
    assert float(back.sum()) == 6.0
    assert np.__file__.startswith(sys.prefix), f"numpy 不在当前环境里: {np.__file__}"
    return f"numpy {np.__version__} <-> torch {torch.__version__} 双向 OK"


@check("matplotlib")
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "mpl.png"
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.4), dpi=80)
    ax[0].imshow(torch.eye(6).numpy(), cmap="viridis")
    ax[1].plot(torch.linspace(0, 1, 16).numpy() ** 2)
    fig.savefig(p)
    plt.close(fig)
    return f"mpl {matplotlib.__version__} 画 tensor OK ({p.stat().st_size // 1024}KB)"


@check("cuda")
def _cuda():
    import torch

    assert torch.cuda.is_available(), "CUDA 不可用（会被静默回退到 CPU，训练慢 20 倍以上）"
    x = torch.randn(256, 256, device="cuda")
    _ = (x @ x).sum().item()
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    return f"{p.name} {total / 1024**3:.1f}GiB cc{p.major}.{p.minor} | 空闲 {free / 1024**2:.0f}MB"


@check("bf16")
def _bf16():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("无 CUDA，bf16 训练不可用（会自动退回 fp32）")
    a = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    _ = (a @ a).float().sum().item()
    return "bf16 matmul OK（Ampere 及以后原生支持）"


@check("attn_backends")
def _backends():
    import torch
    import torch.nn.functional as F

    q = torch.randn(2, 4, 64, 16, device="cuda")
    ok = []
    for name, flags in {
        "flash": dict(enable_flash=True, enable_math=False, enable_mem_efficient=False),
        "mem_efficient": dict(enable_flash=False, enable_math=False, enable_mem_efficient=True),
        "math": dict(enable_flash=False, enable_math=True, enable_mem_efficient=False),
    }.items():
        try:
            with torch.backends.cuda.sdp_kernel(**flags):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
            ok.append(name)
        except Exception:  # noqa: BLE001
            pass
    assert "math" in ok or "mem_efficient" in ok, "没有任何可用的注意力后端"
    return "可用: " + ", ".join(ok) + "（阶段 3 对比须注明所用后端）"


@check("pytest")
def _pytest():
    import pytest

    return f"pytest {pytest.__version__}"


@check("layout")
def _layout():
    for sub in ("src/tlab", "tests", "experiments", "report"):
        assert (HERE / sub).exists(), f"缺少目录 {sub}"
    return f"项目根 {HERE}"


def main() -> int:
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)

    width = max(len(n) for n, _, _ in RESULTS)
    bad = 0
    for name, ok, detail in RESULTS:
        bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")

    pip_mod = Path(__import__("pip").__file__).resolve()
    if not str(pip_mod).startswith(str(Path(sys.prefix).resolve())):
        print(f"\n[note] pip 来自 user-site: {pip_mod}\n"
              f"       装包前先设 $env:PYTHONNOUSERSITE=1，否则可能装到别处。")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("[note] PYTHONNOUSERSITE 未设为 1（run.ps1 已处理；直接用 python 时要注意）")

    print(f"\n{'全部通过' if bad == 0 else f'{bad} 项失败'}  ({len(RESULTS)} 项检查)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
