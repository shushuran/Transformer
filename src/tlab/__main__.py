"""命令行入口：`python -m tlab <子命令>`（推荐用 .\\run.ps1 -m tlab ...）"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from . import config as C
from . import data as D
from .models import build_model, count_params
from .plotting import plot_loss
from .trainer import Trainer


def _parse_sets(pairs: list[str]) -> dict:
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set 需要 k=v 形式，收到 {item!r}")
        k, v = item.split("=", 1)
        for caster in (int, float):
            try:
                out[k] = caster(v)
                break
            except ValueError:
                continue
        else:
            out[k] = {"true": True, "false": False}.get(v.lower(), v)
    return out


def _run_dir(cfg) -> Path:
    p = Path(cfg.dot("run.out_dir")) / str(cfg.dot("run.name"))
    return p if p.is_absolute() else C.ROOT / p


def cmd_env(args) -> int:
    return subprocess.call([sys.executable, str(C.ROOT / "tools" / "check_env.py")])


def cmd_prepare(args) -> int:
    cfg = C.load_config(args.config, **_parse_sets(args.set))
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    C.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    text = D.ensure_corpus(C.DATA_DIR / "raw" / "input.txt", cfg.dot("data.url"))
    tok = D.CharTokenizer.from_text(text, int(cfg.dot("data.min_freq")))
    tok.save(C.DATA_DIR / "tokenizer.json")

    ids = tok.encode(text)
    train_ids, val_ids = D.split_ids(ids, float(cfg.dot("data.val_frac")))
    torch.save(train_ids, C.DATA_DIR / "train.pt")
    torch.save(val_ids, C.DATA_DIR / "val.pt")

    ppl = D.empirical_bigram_ppl(train_ids, val_ids, tok.vocab_size)
    print(f"[prepare] 语料 {len(text):,} 字符 | 词表 {tok.vocab_size} | "
          f"train {len(train_ids):,} / val {len(val_ids):,} | "
          f"统计 bigram PPL={ppl:.3f} | {time.perf_counter() - t0:.1f}s")
    print(f"[prepare] 词表前 12 个 id: {[repr(c) for c in tok.itos[:12]]}")
    print(f"[prepare] 产出: {C.DATA_DIR}")
    return 0


def cmd_stats(args) -> int:
    cfg = C.load_config(args.config, **_parse_sets(args.set))
    tok = D.CharTokenizer.load(C.DATA_DIR / "tokenizer.json")
    train_ids = torch.load(C.DATA_DIR / "train.pt", weights_only=False)
    val_ids = torch.load(C.DATA_DIR / "val.pt", weights_only=False)
    ppl = D.empirical_bigram_ppl(train_ids, val_ids, tok.vocab_size)
    print(f"vocab={tok.vocab_size}  train={len(train_ids):,}  val={len(val_ids):,}")
    print(f"统计 bigram PPL = {ppl:.3f}   <- 神经模型应当与它同量级或更低")
    print(f"均匀分布 PPL   = {tok.vocab_size:.3f}")
    return 0


def cmd_train(args) -> int:
    cfg = C.load_config(args.config, **_parse_sets(args.set))
    C.set_seed(int(cfg["seed"]))

    tok = D.CharTokenizer.load(C.DATA_DIR / "tokenizer.json")
    train_ids = torch.load(C.DATA_DIR / "train.pt", weights_only=False)
    val_ids = torch.load(C.DATA_DIR / "val.pt", weights_only=False)
    cfg["model"]["vocab_size"] = tok.vocab_size

    model = build_model(cfg, tok.vocab_size)
    run_dir = _run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    tr = Trainer(model, cfg, train_ids, val_ids, out_dir=run_dir)
    print(f"[train] model={cfg.dot('model.name')} params={count_params(model):,} "
          f"device={tr.device} amp={'bf16' if tr.use_amp and tr.amp_dtype == torch.bfloat16 else 'off'} "
          f"bs={tr.batch_size}x{tr.grad_accum} block={tr.block_size} steps={tr.max_steps}")
    print(f"[train] 日志: {tr.log_path}")

    t0 = time.perf_counter()
    tr.train()
    png = plot_loss(tr.log_path, run_dir / "loss.png")
    print(f"[train] 总耗时 {time.perf_counter() - t0:.1f}s | 曲线: {png}")
    return 0


def cmd_plot(args) -> int:
    csv_path = Path(args.csv) if args.csv else _run_dir(C.load_config(args.config)) / "log.csv"
    out = plot_loss(csv_path, args.out, logy=args.logy)
    print(f"[plot] {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tlab", description="Transformer 陪练实验室（阶段 0）")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("-c", "--config", default=None, help="实验配置 yaml")
        sp.add_argument("--set", nargs="*", default=[], metavar="K=V", help="临时覆盖配置，如 train.lr=1e-3")

    common(sub.add_parser("env", help="环境门禁自检"))
    common(sub.add_parser("prepare", help="下载语料 + 建词表 + 切分"))
    common(sub.add_parser("stats", help="打印语料统计与 bigram 地板线"))

    st = sub.add_parser("train", help="训练")
    common(st)

    pl = sub.add_parser("plot", help="把 log.csv 画成曲线")
    common(pl)
    pl.add_argument("--csv", default=None)
    pl.add_argument("--out", default="runs/loss.png")
    pl.add_argument("--logy", action="store_true")

    args = p.parse_args(argv)
    return {"env": cmd_env, "prepare": cmd_prepare, "stats": cmd_stats,
            "train": cmd_train, "plot": cmd_plot}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
