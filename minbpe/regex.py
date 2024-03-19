import regex as re
from typing import Dict, List, Tuple, Any, Optional, Union
from .base import Tokenizer, get_stats, merge

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: Optional[str] = None):
        super().__init__()
        self.pattern: str = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern: re.Pattern = re.compile(self.pattern)
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks: List[str] = re.findall(self.compiled_pattern, text)
        ids: List[List[int]] = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges: Dict[Tuple[int, int], int] = {}
        vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats: Dict[Tuple[int, int], int] = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            pair: Tuple[int, int] = max(stats, key=lambda p: stats.get(p, 0))
            idx: int = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx].decode('utf-8', errors='replace')}) had {stats[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens: Dict[str, int]) -> None:
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids: List[int]) -> str:
        part_bytes: List[bytes] = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes: bytes = b"".join(part_bytes)
        text: str = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        ids: List[int] = list(text_bytes)
        while len(ids) >= 2:
            stats: Dict[Tuple[int, int], int] = get_stats(ids)
            pair: Tuple[int, int] = min(
                stats, key=lambda p: self.merges.get(p, float("inf"))
            )
            if pair not in self.merges:
                break
            idx: int = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text: str) -> List[int]:
        text_chunks: List[str] = re.findall(self.compiled_pattern, text)
        ids: List[int] = []
        for chunk in text_chunks:
            chunk_bytes: bytes = chunk.encode("utf-8")
            chunk_ids: List[int] = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(
        self, text: str, allowed_special: Union[str, set, None] = "none_raise"
    ) -> List[int]:
        special: Optional[Dict[str, int]] = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {
                k: v for k, v in self.special_tokens.items() if k in allowed_special
            }
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text)
        special_pattern: str = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks: List[str] = re.split(special_pattern, text)
        ids: List[int] = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
