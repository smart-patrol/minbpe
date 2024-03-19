from typing import Dict, Tuple, List, Any
from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes: bytes = text.encode("utf-8")  # raw bytes
        ids: List[int] = list(text_bytes)  # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges: Dict[Tuple[int, int], int] = {}  # (int, int) -> int
        vocab: Dict[int, bytes] = {
            idx: bytes([idx]) for idx in range(256)
        }  # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats: Dict[Tuple[int, int], int] = get_stats(ids)
            # find the pair with the highest count
            pair: Tuple[int, int] = max(stats, key=lambda p: stats.get(p, 0))
            # mint a new token: assign it the next available id
            idx: int = 256 + i
            # replace all occurrences of pair in ids with idx
            ids: List[int] = merge(ids, pair, idx)  # type: ignore
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx].decode('utf-8', errors='replace')}) had {stats[pair]} occurrences"
                )

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def decode(self, ids: List[int]) -> str:
        # given ids (list of integers), return Python string
        text_bytes: bytes = b"".join(self.vocab[idx] for idx in ids)
        text: str = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> List[int]:
        # given a string text, return the token ids
        text_bytes: bytes = text.encode("utf-8")  # raw bytes
        ids: List[int] = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats: Dict[Tuple[int, int], int] = get_stats(ids)
            pair: Tuple[int, int] = min(
                stats, key=lambda p: self.merges.get(p, float("inf"))
            )  # type: ignore
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx: int = self.merges[pair]
            ids: List[int] = merge(ids, pair, idx)  # type: ignore
        return ids
