"""配置读取 + 全局随机性控制。

阶段 0 的第一条纪律：任何一次实验都必须能被逐位复现。
复现失败最常见的原因不是模型，而是种子和 cudnn 的非确定性。
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

_MISSING = object()

# 所有实验共享的默认值；yaml 里只写和默认值不同的部分
DEFAULTS: dict = {
    "seed": 42,
    "data": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "min_freq": 1,
        "val_frac": 0.1,
        "block_size": 128,
    },
    "model": {"name": "bigram", "d_model": 64},
    "train": {
        "device": "cuda",
        "amp": "bf16",
        "batch_size": 32,
        "grad_accum": 1,
        "max_steps": 2000,
        "lr": 3.0e-3,
        "min_lr_ratio": 0.1,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "betas": [0.9, 0.95],
        "grad_clip": 1.0,
        "log_every": 50,
        "eval_every": 200,
        "eval_batches": 20,
        "ckpt_every": 1000,
    },
    "run": {"name": "stage0", "out_dir": "runs"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """支持点号访问的配置对象：cfg.dot("train.lr")。"""

    def dot(self, key: str, default=_MISSING):
        cur = self
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                if default is _MISSING:
                    raise KeyError(f"配置缺失: {key}")
                return default
            cur = cur[part]
        return cur

    def override(self, **kwargs) -> "Config":
        """命令行覆盖，如 lr=1e-3 -> dot('train.lr')。"""
        for key, val in kwargs.items():
            if val is None:
                continue
            *parents, leaf = key.split(".")
            cur = self
            for p in parents:
                cur = cur.setdefault(p, {})
            cur[leaf] = val
        return self


def load_config(path: str | Path | None = None, **overrides) -> Config:
    raw: dict = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = Config(_deep_merge(DEFAULTS, raw))
    return cfg.override(**overrides)


def set_seed(seed: int = 42) -> None:
    """固定所有随机源，让同一个种子下的两次运行逐位相同。

    必须覆盖四个来源，否则"看起来固定了其实没固定"：
      1) python 内建 random
      2) numpy 全局随机状态（shuffle、数据增强用）
      3) torch CPU 生成器 + 所有 CUDA 生成器
      4) cuDNN 的算法选择（benchmark=False 才不随机挑 kernel）

    提示：
      - os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 必须在 CUDA
        context 建立之前设置，否则部分 GPU matmul 仍不确定；
      - 不要在训练脚本里打开 torch.use_deterministic_algorithms(True)，
        那会让 mem_efficient 注意力反向直接报错（阶段 1/3 会踩到）。
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str = "cuda") -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)
