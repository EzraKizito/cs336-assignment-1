from tokenizer import BPETokenizer
import time

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
for _ in range(50):
    start_time = time.time()
    ts_tokenizer.encode(ts_text)
    end_time = time.time()
    total_time_ts += (end_time - start_time)

print(f"Average TinyStories tokenizer throughput: {len(ts_text.encode("utf-8"))/ (total_time_ts/ 50):.2f} bytes/second\n")

total_time_owt = 0
for _ in range(50):
    start_time = time.time()
    owt_tokenizer.encode(owt_text)
    end_time = time.time()
    total_time_owt += (end_time - start_time)

print(f"Average OWT tokenizer throughput: {len(owt_text.encode("utf-8"))/ (total_time_owt/ 50):.2f} bytes/second\n")