"""字符级语料与批次采样。

为什么用字符级：词表只有几十个，不需要 tokenizer 依赖，模型小到能在
4GB 显存上几分钟跑完一轮 —— 阶段 0/1 的目的是对比机制，不是刷指标。
"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from pathlib import Path

import torch

UNK = "<unk>"


def ensure_corpus(path: Path, url: str, verbose: bool = True) -> str:
    """本地有就读缓存，没有就下载（tinyshakespeare，约 1.1MB）。"""
    path = Path(path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[data] 下载语料 -> {path}")
    req = urllib.request.Request(url, headers={"User-Agent": "tlab/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    text = raw.decode("utf-8")
    path.write_text(text, encoding="utf-8")
    return text


class CharTokenizer:
    """字符 <-> id 的双向映射。id 0 永远是 <unk>，保证编码不会抛异常。"""

    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        if UNK not in self.stoi:
            raise ValueError(f"词表必须包含 {UNK}")
        if len(self.stoi) != len(self.itos):
            raise ValueError("词表里有重复字符")

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @classmethod
    def from_text(cls, text: str, min_freq: int = 1) -> "CharTokenizer":
        """从语料统计字符，构造确定性的词表。

        确定性要求（否则每次运行 id 都不一样，之前训练的 ckpt 就废了）：
          - 按出现频次降序；
          - 频次相同的字符按 unicode 码位升序（不要依赖 Counter 的插入顺序，
            那会随 Python 版本/字典实现变化）；
          - <unk> 固定放在 0 号位。

        min_freq 用来过滤极稀有字符，把它们统一映射到 <unk>。
        """
        counts = Counter(text)
        items = [(ch, n) for ch, n in counts.items() if n >= min_freq and ch != UNK]
        items.sort(key=lambda kv: (-kv[1], ord(kv[0])))
        return cls([UNK] + [ch for ch, _ in items])

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def to_json(self) -> dict:
        return {"itos": self.itos}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["itos"])


def split_ids(ids: list[int] | torch.Tensor, val_frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """尾部连续切出验证集（不随机打乱，避免与训练集重叠泄漏）。"""
    t = torch.as_tensor(ids, dtype=torch.long)
    n_val = max(1, int(len(t) * val_frac))
    return t[:-n_val], t[-n_val:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机采 batch_size 个长度为 block_size 的连续片段。

    返回 (x, y)，形状都是 (batch_size, block_size)，dtype=torch.long。
    y 是 x 在语料里各自右移一位的结果 —— 也就是"用前 block_size 个字预测
    紧接着的下一个字"。注意 y 必须从 x 的同一窗口错一位取，而不是另采一批：
    否则模型学到的是错位的目标。

    generator 传进来时，结果必须完全由它决定（测试会检查这一点）。
    """
    data = torch.as_tensor(data, dtype=torch.long)
    if block_size < 1:
        raise ValueError("block_size 必须 >= 1")
    if len(data) <= block_size + 1:
        raise ValueError(f"语料太短: {len(data)} <= block_size+1={block_size + 1}")
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start, (batch_size,), generator=generator)
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])
    return x, y


def empirical_bigram_ppl(
    train_ids: torch.Tensor, val_ids: torch.Tensor, vocab_size: int, alpha: float = 1.0
) -> float:
    """加一平滑的字级 bigram 模型在验证集上的困惑度。

    这是阶段 0 的"地板线"：神经模型如果连它都打不过，说明训练管线有问题。
    """
    t = torch.as_tensor(train_ids, dtype=torch.long)
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.float64)
    counts.index_put_((t[:-1], t[1:]), torch.ones(len(t) - 1, dtype=torch.float64), accumulate=True)
    probs = (counts + alpha) / (counts.sum(dim=1, keepdim=True) + alpha * vocab_size)

    v = torch.as_tensor(val_ids, dtype=torch.long)
    nll = -torch.log(probs[v[:-1], v[1:]])
    return float(torch.exp(nll.mean()))
