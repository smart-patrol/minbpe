"""
Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer.
Note that this is a pretrained tokenizer. By default and inside init(), it
loads the pretrained tokenizer from the `cl100k_base` tokenizer of tiktoken.
"""

from typing import Dict, Tuple, List, Optional
import tiktoken
from .regex import RegexTokenizer


def bpe(
    mergeable_ranks: Dict[bytes, int], token: bytes, max_rank: Optional[int] = None
) -> List[bytes]:
    parts: List[bytes] = [bytes([b]) for b in token]
    while True:
        min_idx: Optional[int] = None
        min_rank: Optional[int] = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank: Optional[int] = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2 :]
        )
    return parts


def recover_merges(mergeable_ranks: Dict[bytes, int]) -> Dict[Tuple[int, int], int]:
    merges: Dict[Tuple[int, int], int] = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue  # skip raw bytes
        pair: Tuple[bytes, ...] = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0: int = mergeable_ranks[pair[0]]
        ix1: int = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


class GPT4Tokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""

    def __init__(self) -> None:
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks: Dict[bytes, int] = enc._mergeable_ranks
        self.merges: Dict[Tuple[int, int], int] = recover_merges(mergeable_ranks)
        vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        self.byte_shuffle: Dict[int, int] = {
            i: mergeable_ranks[bytes([i])] for i in range(256)
        }
        self.inverse_byte_shuffle: Dict[int, int] = {
            v: k for k, v in self.byte_shuffle.items()
        }
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids: List[int] = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids: List[int]) -> str:
        text_bytes: bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text: str = text_bytes.decode("utf-8", errors="replace")
        return text

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError

    def save(self, file_prefix: str) -> None:
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file: str) -> None:
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")

    def save_vocab(self, vocab_file: str) -> None:
        from .base import render_token

        vocab: Dict[int, bytes] = {
            idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)
        }
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        inverted_merges: Dict[int, Tuple[int, int]] = {
            idx: pair for pair, idx in self.merges.items()
        }
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s: str = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0: str = render_token(vocab[idx0])
                    s1: str = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
