"""训练器：AMP(bf16) + 梯度累积 + 梯度裁剪 + warmup/cosine + 日志 + 断点续训。

这些是综述里不会写、但真实项目每天都要用的东西。阶段 0 把它们一次做对，
后面每个实验就只需要换模型和配置。
"""

from __future__ import annotations

import contextlib
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import resolve_device
from .data import get_batch
from .schedule import lr_at

LOG_FIELDS = ["step", "lr", "train_loss", "val_loss", "val_ppl", "tokens", "tok_per_s", "elapsed_s"]


class Trainer:
    def __init__(self, model, cfg, train_ids, val_ids, out_dir: str | Path = "runs/stage0"):
        self.cfg = cfg
        self.device = resolve_device(cfg.dot("train.device"))
        self.model = model.to(self.device)

        t = cfg["train"]
        self.block_size = int(cfg.dot("data.block_size"))
        self.batch_size = int(t["batch_size"])
        self.grad_accum = int(t["grad_accum"])
        self.max_steps = int(t["max_steps"])
        self.base_lr = float(t["lr"])
        self.min_lr_ratio = float(t["min_lr_ratio"])
        self.warmup_steps = int(t["warmup_steps"])
        self.grad_clip = float(t["grad_clip"]) if t.get("grad_clip") else 0.0
        self.log_every = int(t["log_every"])
        self.eval_every = int(t["eval_every"])
        self.eval_batches = int(t["eval_batches"])
        self.ckpt_every = int(t["ckpt_every"])

        self.amp_mode = str(t["amp"]).lower()
        self.use_amp = self.amp_mode in ("bf16", "fp16") and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.amp_mode == "bf16" else torch.float16

        # 只对 >=2 维的参数做 weight decay（LayerNorm/偏置不衰减）
        decay, no_decay = [], []
        for p in self.model.parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": float(t["weight_decay"])},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.base_lr,
            betas=tuple(float(b) for b in t["betas"]),
        )

        self.train_ids = torch.as_tensor(train_ids, dtype=torch.long)
        self.val_ids = torch.as_tensor(val_ids, dtype=torch.long)
        self.vocab_size = int(cfg.dot("model.vocab_size", self.train_ids.max().item() + 1))
        self.gen = torch.Generator().manual_seed(int(cfg["seed"]))

        self.step = 0
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "log.csv"
        self._rows: list[dict] = []
        self._csv_ready = False

    # ---------------------------------------------------------------- helpers
    def _amp(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    def _log(self, **row) -> None:
        self._rows.append(row)
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in LOG_FIELDS})

    # -------------------------------------------------------------- evaluate
    def evaluate(self, n_batches: int | None = None) -> dict:
        """在验证集上算平均交叉熵和困惑度。

        约定（测试会检查）：
          - 全程 torch.no_grad()；
          - 用固定种子的独立 generator 采验证批次，所以调用两次结果必须完全一致，
            否则曲线上的抖动会被误读成模型在变化；
          - 结束时把模型恢复回训练模式，不能污染训练循环；
          - 返回 {"val_loss": float, "val_ppl": float, "val_batches": int}，
            其中 val_ppl == exp(val_loss)；batch 数取 min(n_batches, 可用批次数)。
        """
        n = int(n_batches if n_batches is not None else self.eval_batches)
        max_batches = (len(self.val_ids) - 1) // self.block_size
        n = max(1, min(n, max_batches))

        gen = torch.Generator().manual_seed(int(self.cfg["seed"]) + 9999)
        was_training = self.model.training
        self.model.eval()
        total = 0.0
        try:
            with torch.no_grad():
                for _ in range(n):
                    x, y = get_batch(self.val_ids, self.batch_size, self.block_size, gen)
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    with self._amp():
                        logits = self.model(x)
                        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
                    total += float(loss.detach())
        finally:
            if was_training:
                self.model.train()
        mean_loss = total / n
        return {"val_loss": mean_loss, "val_ppl": math.exp(mean_loss), "val_batches": n}

    # ----------------------------------------------------------------- train
    def train(self) -> list[dict]:
        cfg = self.cfg
        t = cfg["train"]
        total = self.max_steps
        accum = self.grad_accum
        block, bs = self.block_size, self.batch_size

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        micro = 0
        last_loss = float("nan")
        t0 = time.perf_counter()
        tokens = 0
        start = self.step

        while self.step < total:
            lr = lr_at(self.step, self.base_lr, total, self.warmup_steps, self.min_lr_ratio)
            for g in self.opt.param_groups:
                g["lr"] = lr

            x, y = get_batch(self.train_ids, bs, block, self.gen)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with self._amp():
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))

            (loss / accum).backward()
            micro += 1
            tokens += x.numel()

            if micro == accum:
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.step += 1
                micro = 0
                last_loss = float(loss.detach())
                elapsed = time.perf_counter() - t0

                row = dict(
                    step=self.step,
                    lr=round(lr, 8),
                    train_loss=round(last_loss, 5),
                    tokens=tokens,
                    tok_per_s=round(tokens / max(elapsed, 1e-9)),
                    elapsed_s=round(elapsed, 2),
                )
                if self.eval_every and self.step % self.eval_every == 0:
                    ev = self.evaluate()
                    row.update(val_loss=round(ev["val_loss"], 5), val_ppl=round(ev["val_ppl"], 4))
                    print(
                        f"[step {self.step:>5}/{total}] lr={lr:.2e} "
                        f"train={last_loss:.4f} val={ev['val_loss']:.4f} ppl={ev['val_ppl']:.3f} "
                        f"{row['tok_per_s']:,} tok/s",
                        flush=True,
                    )
                elif self.step % self.log_every == 0:
                    print(
                        f"[step {self.step:>5}/{total}] lr={lr:.2e} train={last_loss:.4f} "
                        f"{row['tok_per_s']:,} tok/s",
                        flush=True,
                    )
                self._log(**row)

                if self.ckpt_every and self.step % self.ckpt_every == 0:
                    self.save_ckpt(self.out_dir / f"ckpt_{self.step}.pt")

        ev = self.evaluate()
        self._log(
            step=self.step, lr=lr_at(total, self.base_lr, total, self.warmup_steps, self.min_lr_ratio),
            train_loss=round(last_loss, 5), val_loss=round(ev["val_loss"], 5),
            val_ppl=round(ev["val_ppl"], 4), tokens=tokens,
            tok_per_s=round(tokens / max(time.perf_counter() - t0, 1e-9)),
            elapsed_s=round(time.perf_counter() - t0, 2),
        )
        self.save_ckpt(self.out_dir / "last.pt")
        print(f"[done] {self.step - start} 步, 最终 val_ppl={ev['val_ppl']:.3f}, "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        return self._rows

    # ------------------------------------------------------------ checkpoint
    def save_ckpt(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "step": self.step,
                "runner_gen": self.gen.get_state(),
                "config": dict(self.cfg),
            },
            path,
        )

    def load_ckpt(self, path: str | Path) -> None:
        ck = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.step = int(ck["step"])
        if "runner_gen" in ck:
            self.gen.set_state(ck["runner_gen"])
