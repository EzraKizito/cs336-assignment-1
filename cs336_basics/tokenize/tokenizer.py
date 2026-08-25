from cs336_basics.tokenize.train_bpe import PRETOKENIZATION_PATTERN

from typing import Optional, Iterable, Iterator, Any
from collections import Counter

import pickle
import re

class BPETokenizer: 
    def __init__(
            self, 
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: Optional[list[str]] = None
        ) -> None:
        self.vocab = vocab 
        self.inverted_vocab = {v: k for k, v in self.vocab.items()} # for easy lookup
        self.merges = merges
        self.merges_map = {merge: pos for pos, merge in enumerate(self.merges)}
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: Optional[list[str]] = None
    ): 
        with open(vocab_filepath, 'rb') as file: 
            vocab = pickle.load(file)

        with open(merges_filepath, 'rb') as file: 
                merges = pickle.load(file)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]: 
        # Pretokenize text; see pretokenize method. Result is a dictionary that has every 
        # single pretoken (including special tokens) as a key and its special identity as the value
        pretokens = self.pretokenize(text)

        encoded_list = []
        for pretoken, special_identity in pretokens: 
            if special_identity: 
                # look it up in vocab directly
                pretoken_bytes = bytes(pretoken.encode("utf-8"))
                key = self.inverted_vocab[pretoken_bytes]
                encoded_list.append(key) 
            else: 
                token_split = self.convert_pretoken_to_tokens(pretoken=pretoken)

                encoded_text = [self.inverted_vocab[p] for p in token_split]
                encoded_list.extend(encoded_text)
                
        return encoded_list
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable: 
            pretokens = self.pretokenize(text)

            for pretoken, special_identity in pretokens: 
                if special_identity: 
                    # look it up in vocab directly
                    pretoken_bytes = bytes(pretoken.encode("utf-8"))
                    key = self.inverted_vocab[pretoken_bytes]
                    yield key
                else: 
                    token_split = self.convert_pretoken_to_tokens(pretoken=pretoken)

                    for p in token_split: 
                        yield self.inverted_vocab[p]

    def decode(self, ids: list[int]) -> str: 
        # For each id in the list, get vocab entry or unicode entry
        to_bytes = [self.vocab.get(id, '\uFFFD'.encode('utf-8')) for id in ids]
        byte_sequence = b''.join(to_bytes)
        decoded = byte_sequence.decode("utf-8", errors='replace')
        return decoded

    def pretokenize(self, text: str) -> list[tuple[str, bool]]: 
        # First, handle special tokens; we don't want to merge them, 
        # so we identify all the places they begin
        splits = []
        if self.special_tokens:
            escaped = []
            for s in sorted(self.special_tokens, key=len, reverse=True):
                escaped.append(re.escape(s))
            split_delim = f"({"|".join(escaped)})"
            splits: list[str] = re.split(split_delim, text)
        else: 
            splits = [text]

        pretokens_list = []
        for s in splits: 
            if self.special_tokens:
                if s in self.special_tokens: 
                    pretokens_list.append((s, True))
                    continue
            pretokens = PRETOKENIZATION_PATTERN.findall(s)
            for p in pretokens: 
                pretokens_list.append((p, False))

        # A list of tuples, each tuple having the pretoken and a special token flag.
        return pretokens_list

    def convert_pretoken_to_tokens(self, pretoken: str) -> list[bytes]: 
        split = [bytes([b]) for b in pretoken.encode("utf-8")]

        # while loop to do all the merges within a pretoken until there's none left
        while True: 
            new_split: list[bytes] = []
            pairs = [(split[j], split[j+1]) for j in range(len(split) - 1)] # Look at all adjacent pairs
            merge_positions = [self.merges_map.get(p) for p in pairs]

            # if merge positions are none, we break
            if all(pos is None for pos in merge_positions):
                break

            i = 0
            first_merge_pos = min([pos for pos in merge_positions if pos is not None]) # Find position of first merge
            while i < len(split):
                if i < len(split) - 1:
                    if merge_positions[i] == first_merge_pos:
                        new_token = split[i] + split[i+1]
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                else: 
                    new_split.append(split[i])
                    i += 1
            if new_split == split: # Stopping condition; reconstructing doesn't change the split
                break
            split = new_split

        return split
