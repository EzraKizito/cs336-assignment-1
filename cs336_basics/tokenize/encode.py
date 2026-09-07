from typing import Literal
from tokenizer import BPETokenizer
from train_bpe import find_chunk_boundaries, get_folder_size_pathlib, format_byte_size, format_time_duration

import os
import time
import numpy as np

ts_merges = "trained_tokenizer/tinystories/merges.pkl"
ts_vocab = "trained_tokenizer/tinystories/vocab.pkl"
owt_merges = "trained_tokenizer/owt/merges.pkl"
owt_vocab = "trained_tokenizer/owt/vocab.pkl"

special_tokens = ["<|endoftext|>"]
# Initialize tokenizers
ts_tokenizer = BPETokenizer.from_files(
    vocab_filepath=ts_vocab, 
    merges_filepath=ts_merges,
    special_tokens=special_tokens
)

owt_tokenizer = BPETokenizer.from_files(
    vocab_filepath=owt_vocab, 
    merges_filepath=owt_merges,
    special_tokens=special_tokens
)

# Encode function
def encode(
        dataset: Literal["tinystories", "owt"], 
        dataset_type: Literal["train", "test"]
    ):
    if dataset == "owt":
        tokenizer = owt_tokenizer
        if dataset_type == "train":
            filepath = "data/owt_train.txt"
            output_path = "data/encoded/owt.dat"
        else:
            filepath = "data/owt_valid.txt"
            output_path = "data/encode/owt_valid.dat"
    else: 
        tokenizer = ts_tokenizer
        if dataset_type == "train":
            filepath = "data/TinyStoriesV2-GPT4-train.txt"
            output_path = "data/encoded/tinystories.dat"
        else: 
            filepath = "data/TinyStoriesV2-GPT4-valid.txt"
            output_path = "data/encoded/tinystories_valid.dat"

    # While file is open, chunk into boundaries separated by about 100 MB
    with open(filepath, 'rb') as f: 
        file_size = os.fstat(f.fileno()).st_size
        num_chunks = -(-file_size // (10*1024*1024))
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    # Process each chunk lazily
    with open(output_path, 'wb') as file:
        with open(filepath, 'rb') as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8")

                # TODO: Look into np.fromiter for a more memory efficient implementation
                ids = []
                for id_sequence in tokenizer.encode_iterable([chunk]):
                    ids.append(id_sequence)
                numpy_ids = np.array(ids, dtype=np.uint16)

                numpy_ids.tofile(file)

    return output_path

def encode_tinystories(split: Literal["train", "test"]):
    print(f"Encoding TinyStories ({split})\n")

    ts_st = time.time()
    ts_arr_output_path = encode("tinystories", dataset_type=split)
    ts_et = time.time()
    print(f"Tiny Stories Encoded Array Size: {format_byte_size(get_folder_size_pathlib(ts_arr_output_path))}\n")
    print(f"Tiny Stories encoding took {format_time_duration(ts_et - ts_st)}")

def encode_owt(split: Literal["train", "test"]): 
    print("Encoding OWT ({split})\n")
    owt_st = time.time()
    owt_arr_output_path = encode("owt", dataset_type=split)
    owt_et = time.time()
    print(f"OWT Encoded Array Size: {format_byte_size(get_folder_size_pathlib(owt_arr_output_path))}\n")
    print(f"OWT encoding took {format_time_duration(owt_et - owt_st)}")

# encode_tinystories("train")
encode_tinystories("test")
# encode_owt("train")
encode_owt("test")