"""学习率调度：线性 warmup + cosine 衰减。

为什么单独成模块：它是阶段 1「Post-LN 不加 warmup 会发散」实验的自变量。
把它写成一个纯函数，实验里才好一次只改一个参数。
"""

from __future__ import annotations

import math


def lr_at(
    step: int,
    base_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    """返回第 step 次优化器更新时使用的学习率（step 从 0 开始）。

    分段定义（照这个写，测试就是照它断言的）：
      1) warmup 段：step < warmup_steps 时
             lr = base_lr * (step + 1) / warmup_steps
         —— 注意 step=0 时是 base_lr/warmup_steps 而不是 0，
            否则第一步更新为 0，白白浪费一次前向。
      2) 衰减段：warmup_steps <= step < total_steps 时
             progress = (step - warmup_steps) / (total_steps - warmup_steps)
             lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
         其中 min_lr = base_lr * min_lr_ratio。
      3) 收尾：step >= total_steps 时 lr = min_lr。

    边界：warmup_steps = 0 时第 0 步直接取 base_lr（进度 0 -> cos=1）。
    """
    min_lr = base_lr * min_lr_ratio
    if step >= total_steps:
        return min_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
