import os
import sys
import pickle
import time
from collections import Counter, defaultdict
from itertools import repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import BinaryIO, Literal

import regex as re

PRETOKENIZATION_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _remove_special_tokens(special_tokens: list[str], chunk: str) -> list[str]: 
    escaped = []
    for s in special_tokens: 
        escaped.append(re.escape(s))
    split_delim = "|".join(escaped)
    splits = re.split(split_delim, chunk)
    return splits
       
def process_bound_pretokenization(
    start: int, 
    end: int, 
    file_path: str, 
    special_tokens: list[str]
) -> Counter: 
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
        splits = _remove_special_tokens(special_tokens, chunk)

        count_mapping = Counter()
        for s in splits: 
            pretokens = PRETOKENIZATION_PATTERN.findall(s)
            count_mapping.update(pretokens)
        
        return count_mapping 
    
def optimized_train_bpe_tokenizer(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str], 
    multiprocess: bool = False
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
    base_dir = Path(__file__).resolve().parent.parent
    file_path = os.path.join(base_dir, input_path)
    
    # Pretokenization step
    master_counter = Counter()

    print("\nChunking file...\n")
    pretoken_start_time = time.time()
    with open(file_path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        num_chunks = cpu_count() if multiprocess else -(-file_size // (100*1024*1024))
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
    
    if multiprocess:
        with Pool() as pool: 
            results = pool.starmap(process_bound_pretokenization, zip(boundaries[:-1], boundaries[1:], repeat(file_path), repeat(special_tokens)))

        for child_counter in results: 
            master_counter.update(child_counter)
    else: 
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            child_counter = process_bound_pretokenization(start, end, file_path, special_tokens)
            master_counter.update(child_counter)

    pretoken_end_time = time.time()
    print(f"\nPretokenization took {format_time_duration(pretoken_end_time - pretoken_start_time)}")
                                        
    # Initialize vocabulary
    vocabulary = {}
    for i in range(256):
        vocabulary[i] = bytes([i])
    
    for token in special_tokens: 
        vocabulary[len(vocabulary)] = token.encode("utf-8")
    
    # Initialize word split tracker AND inverted tracker
    word_splits = {}
    word_counts = Counter()
    pair_counts = Counter()
    inverted_index = defaultdict(set)

    bk_st = time.time()
    # We do all the bookkeeping before the loop
    for word_id, (word, count) in enumerate(master_counter.items()):
        split = [bytes([b]) for b in word.encode("utf-8")]
        word_splits[word_id] = split
        word_counts[word_id] = count

        for i in range(len(split) - 1): 
            pair = (split[i], split[i+1])
            pair_counts[pair] += count
            inverted_index[pair].add(word_id)

    bk_et = time.time()
    print(f"Creating counts took {format_time_duration(bk_et - bk_st)}\n")
    print(f"Word split size: {format_byte_size(sys.getsizeof(word_splits))}\n")
    print(f"Inverted index size: {format_byte_size(sys.getsizeof(inverted_index))}\n")
    print(f"Pair Count size: {format_byte_size(sys.getsizeof(pair_counts))}\n")
    merges = []

    print("\nAdding to vocabulary...\n")
    vocab_st = time.time()
    while len(vocabulary) < vocab_size: 
        if not pair_counts:
            break
        
        # Pick lexicographically greater pair
        top_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))

        new_merge = top_pair[0]
        
        # Record merge and add it to vocabulary
        merges.append(new_merge)
        vocabulary[len(vocabulary)] = new_merge[0] + new_merge[1]
        
        # Apply the merge only in the affected words
        affected_words = list(inverted_index[new_merge]) 
        del pair_counts[new_merge]
        del inverted_index[new_merge]

        # We recompute counts only for the affected words
        for word_id in affected_words: 
            split = word_splits[word_id]
            count = word_counts[word_id]

            # Remove old split counts and remove pair from word ID for all pairs
            for i in range(len(split) - 1): 
                p = (split[i], split[i+1])
                pair_counts[p] -= count 
                inverted_index[p].discard(word_id)

            # Merge and build new splits in place
            new_split = []
            i = 0 
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == new_merge: 
                    new_token = split[i] + split[i+1]
                    new_split.append(new_token)
                    i += 2
                else: 
                    new_split.append(split[i])
                    i += 1
           
            word_splits[word_id] = new_split

            # Update split counts
            for i in range(len(new_split) -  1): 
                p = (new_split[i], new_split[i+1])
                pair_counts[p] += count 
                inverted_index[p].add(word_id)

        if len(vocabulary) % 1000 == 0: 
            now = time.time()
            print(f"Vocab size {len(vocabulary)} achieved {format_time_duration(now - vocab_st)} after beginning vocab add time.")

    vocab_et = time.time()
    print(f"\nVocabulary expansion took {format_time_duration(vocab_et - vocab_st)}")

    return vocabulary, merges

def get_folder_size_pathlib(folder_path):
    root = Path(folder_path)
    if root.is_file():
        return root.stat().st_size
    # Recursively match all files using rglob
    return sum(f.stat().st_size for f in root.rglob('*') if f.is_file())

def format_byte_size(size: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"  

def format_time_duration(seconds: float) -> str:
    # Handle zero or sub-second values gracefully
    if seconds < 1:
        return f"{seconds:.2f}s" if seconds > 0 else "0s"

    seconds = int(seconds)
    
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)

def train_bpe_runner(dataset_name: Literal["tinystories", "owt"]) -> tuple[float, int]: 
    file_path = "data/TinyStoriesV2-GPT4-train.txt" if dataset_name == "tinystories" else "data/owt_train.txt"
    vocab_size = 10000 if dataset_name == "tinystories" else 32000
    special_tokens = ["<|endoftext|>"]
    multiprocess = True if dataset_name == "tinystories" else False

    start_time = time.time()
    vocab, merges = optimized_train_bpe_tokenizer(
        input_path=file_path, 
        vocab_size=vocab_size, 
        special_tokens=special_tokens, 
        multiprocess=multiprocess
    )
    end_time = time.time()
    duration = end_time - start_time
    os.makedirs(f"trained_tokenizer/{dataset_name}", exist_ok=True)

    with open(f"trained_tokenizer/{dataset_name}/vocab.pkl", "wb") as f: 
        pickle.dump(vocab, f)
    with open(f"trained_tokenizer/{dataset_name}/merges.pkl", "wb") as f: 
        pickle.dump(merges, f)

    size_in_bytes = get_folder_size_pathlib(f"trained_tokenizer/{dataset_name}")
    
    return duration, size_in_bytes

if __name__ == '__main__':
    # Multithreading best practices
    import os
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    for name in ("tinystories", "owt"): 
        print(f"\n############ TRAINING ON {name} #################\n")
        print(f"\nBeginning training tokenizer on {name} Dataset...\n")

        duration, size_in_bytes = train_bpe_runner(name)
        print(f"\n## SUMMARY: {name}\n")
        print(f"\nTraining BPE tokenizer on {name} took {format_time_duration(duration)}")
        print(f"Memory usage of tokenizer on {name} training: {format_byte_size(size_in_bytes)}")


    