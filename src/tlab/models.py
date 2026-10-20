"""阶段 0 的对照模型：低秩分解的字级 bigram。

它不是 Transformer，作用有两个：
  1) 把训练管线（采样/AMP/累积/日志/ckpt）跑通，出问题能立刻定位到管线而不是模型；
  2) 给出一个"地板线"：神经模型的 PPL 应该和加一平滑的统计 bigram PPL 很接近，
     差得远就说明你的实现或超参有问题。

阶段 1 会用自己写的 Transformer 块替换它。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BigramLM(nn.Module):
    """logits_t = W (E_cur[x_t] + E_prev[x_{t-1}])，等价于低秩分解的 bigram 表。"""

    def __init__(self, vocab_size: int, d_model: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.cur = nn.Embedding(vocab_size, d_model)
        self.prev = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) long -> logits: (B, T, V)
        prev = torch.cat([torch.zeros_like(idx[:, :1]), idx[:, :-1]], dim=1)
        return self.head(self.cur(idx) + self.prev(prev))


def build_model(cfg, vocab_size: int) -> nn.Module:
    name = cfg.dot("model.name")
    if name == "bigram":
        return BigramLM(vocab_size, int(cfg.dot("model.d_model")))
    raise ValueError(f"阶段 0 只有 bigram 基线，收到 model.name={name!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
