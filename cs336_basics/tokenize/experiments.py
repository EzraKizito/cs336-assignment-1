from tokenizer import BPETokenizer
from train_bpe import find_chunk_boundaries, get_folder_size_pathlib, format_byte_size, format_time_duration

import time
from typing import Literal
import os
import numpy as np

ts_sample_path = "data/samples/tinystories-sample.txt"
owt_sample_path = "data/samples/owt-samples.txt"

with open(ts_sample_path, 'r') as f:
    ts_text = f.read()
with open(owt_sample_path, 'r') as f:
    owt_text = f.read()

ts_merges = "trained_tokenizer/tinystories/merges.pkl"
ts_vocab = "trained_tokenizer/tinystories/vocab.pkl"
owt_merges = "trained_tokenizer/owt/merges.pkl"
owt_vocab = "trained_tokenizer/owt/vocab.pkl"

special_tokens = ["<|endoftext|>"]
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
### Dataset specific compression ratio
ts_encoded_samples = ts_tokenizer.encode(ts_text)
owt_encoded_samples = owt_tokenizer.encode(owt_text)

ts_cr = len(ts_text.encode("utf-8")) / len(ts_encoded_samples)
owt_cr = len(owt_text.encode("utf-8")) / len(owt_encoded_samples)

print(f"Tiny Stories Compression Ratio: {ts_cr:.2f} bytes/token\n")
print(f"OpenWebText Compression Ratio: {owt_cr:.2f} bytes/token\n")

### Switch Encoders -> Compression ratio goes down 
wrongly_encoded_ts = owt_tokenizer.encode(ts_text)
wrongly_encoded_owt = ts_tokenizer.encode(owt_text)

wrong_ts_cr = len(ts_text.encode("utf-8")) / len(wrongly_encoded_ts)
wrong_owt_cr = len(owt_text.encode("utf-8")) / len(wrongly_encoded_owt)

print(f"(Wrongly Encoded) Tiny Stories Compression Ratio: {wrong_ts_cr:.2f} bytes/token\n")
print(f"(Wrongly Encoded) OpenWebText Compression Ratio: {wrong_owt_cr:.2f} bytes/token\n")
#### ESTIMATE THROUGHPUT
total_time_ts = 0
for _ in range(5):
    start_time = time.time()
    ts_tokenizer.encode(ts_text)
    end_time = time.time()
    total_time_ts += (end_time - start_time)

print(f"Average TinyStories tokenizer throughput: {len(ts_text.encode("utf-8"))/ (total_time_ts/ 5):.2f} bytes/second\n")

total_time_owt = 0
for _ in range(5):
    start_time = time.time()
    owt_tokenizer.encode(owt_text)
    end_time = time.time()
    total_time_owt += (end_time - start_time)

print(f"Average OWT tokenizer throughput: {len(owt_text.encode("utf-8"))/ (total_time_owt/ 5):.2f} bytes/second\n")

# Encode function
def encode(dataset: Literal["tinystories", "owt"]):
    if dataset == "owt":
        filepath = "data/owt_train.txt"
        tokenizer = owt_tokenizer
        output_path = "data/encoded/owt.dat"
    else: 
        filepath = "data/TinyStoriesV2-GPT4-train.txt"
        tokenizer = ts_tokenizer
        output_path = "data/encoded/tinystories.dat"

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

print("Encoding TinyStories\n")
ts_st = time.time()
ts_arr_output_path = encode("tinystories")
ts_et = time.time()
print(f"Tiny Stories Encoded Array Size: {format_byte_size(get_folder_size_pathlib(ts_arr_output_path))}\n")
print(f"Tiny Stories encoding took {format_time_duration(ts_et - ts_st)}")


print("Encoding OWT\n")
owt_st = time.time()
owt_arr_output_path = encode("owt")
owt_et = time.time()
print(f"OWT Encoded Array Size: {format_byte_size(get_folder_size_pathlib(owt_arr_output_path))}\n")
print(f"OWT encoding took {format_time_duration(owt_et - owt_st)}")
