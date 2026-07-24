import os
from typing import BinaryIO
import regex as re
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from itertools import repeat
import time
import pickle

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
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        splits = _remove_special_tokens(special_tokens, chunk)
        
        pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens = []
        for s in splits: 
            for p in re.finditer(pretokenization_pattern, s): 
                pretokens.append(p.group())
                    
        count_mapping = Counter(pretokens)
        return count_mapping 
    
def train_bpe_tokenizer(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str], 
    multiprocess: bool = True
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
    base_dir = Path(__file__).resolve().parent.parent
    file_path = os.path.join(base_dir, input_path)
    
    # Pretokenization step
    master_counter = Counter()


    with open(file_path, "rb") as f:
        num_processes = cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        if multiprocess:
            with Pool(processes=num_processes) as pool: 
                results = pool.starmap(process_bound_pretokenization, zip(boundaries[:-1], boundaries[1:], repeat(file_path), repeat(special_tokens)))

            for child_counter in results: 
                master_counter.update(child_counter)
        else: 
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                splits = _remove_special_tokens(special_tokens, chunk)
        
                pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                pretokens = []
                for s in splits: 
                    for p in re.finditer(pretokenization_pattern, s): 
                        pretokens.append(p.group())
                    
                count_mapping = Counter(pretokens)
                master_counter.update(count_mapping)

    # Initialize vocabulary
    vocabulary = {}
    for i in range(256):
        vocabulary[i] = bytes([i])
    
    special_tokens = ["<|endoftext|>"]
    for token in special_tokens: 
        vocabulary[len(vocabulary)] = token.encode("utf-8")
    
    # Initial word split tracker
    word_splits = {}
    for word, count in master_counter.items():
        byte_list = [bytes([b]) for b in word.encode("utf-8")]
        word_splits[word] = byte_list
        
    merges = []
    
    while len(vocabulary) < vocab_size: 
        # Reinitialize pair counter
        pair_counts = Counter()
        
        # Count pairs across all word splits
        for word, split in word_splits.items(): 
            pairs = [(split[i], split[i+1]) for i in range(len(split) - 1)]
        
            for pair in pairs: 
                pair_counts[pair] += master_counter[word] 
        
        # Pick lexicographically greater pair
        top_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        
        # Record merge and add it to vocabulary
        new_merge = top_pair[0]
        merges.append(new_merge)
        vocabulary[len(vocabulary)] = top_pair[0][0] + top_pair[0][1]
        
        # Apply the merge everywhere
        for word, split in word_splits.items(): 
            pairs = [(split[i], split[i+1]) for i in range(len(split) - 1)]
            if new_merge not in pairs:
                continue
            
            new_word_splits = []
            i = 0
            
            while i < len(split) - 1:
                if (split[i], split[i + 1]) == new_merge: 
                    new_token = split[i] + split[i+1]
                    new_word_splits.append(new_token)
                    i += 2
                else: 
                    new_word_splits.append(split[i])
                    i += 1
            
            # Add last one if not merged. 
            if i != len(split): 
                new_word_splits.append(split[i])
                
            word_splits[word] = new_word_splits
    
    return vocabulary, merges

def get_folder_size_pathlib(folder_path):
    root = Path(folder_path)
    # Recursively match all files using rglob
    return sum(f.stat().st_size for f in root.rglob('*') if f.is_file())

def format_byte_size(size: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"  

def format_time_duration(seconds: float) -> str:
    units = [("s", 60.0), ("m", 60.0), ("h", 24.0), ("d", 365.0)]

    val = float(seconds)
    for unit, step in units:
        if abs(val) < step:
            return f"{val:.2f} {unit}"
        val /= step

    return f"{val:.2f} y" 

if __name__ == '__main__':
    # Multithreading best practices
    import os
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # # Training on TinyStories
    # print("\n############ TRAINING ON TINY STORIES #################\n")
    # print("\nBeginning training tokenizer on Tiny Stories Dataset...\n")
    # start_time = time.time()

    # tiny_stories_vocabulary, tiny_stories_merges = train_bpe_tokenizer(
    #     "data/TinyStoriesV2-GPT4-train.txt", 
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"]
    # )
    # end_time = time.time()
    # print(f"\nTraining BPE tokenizer on tiny stories took {format_time_duration(end_time - start_time)}\n")

    # print(f"\nSaving to Disk...\n")
    # save_start_time = time.time()
    # os.makedirs("artifacts/tokenizer/tinystories", exist_ok=True)
    # with open("artifacts/tokenizer/tinystories/vocab.pkl", "wb") as f: 
    #     pickle.dump(tiny_stories_vocabulary, f)
    
    # with open("artifacts/tokenizer/tinystories/merges.pkl", "wb") as f: 
    #     pickle.dump(tiny_stories_merges, f)
    # save_end_time = time.time()
    # print(f"\n Saving to disk took {format_time_duration(save_end_time - save_start_time)}\n")

    # size_in_bytes = get_folder_size_pathlib("artifacts/tokenizer/tinystories")
    # print(f"Memory usage of tokenizer on TinyStories: {format_byte_size(size_in_bytes)}")

    # Training on Open Web Text 
    print("\n############ TRAINING ON OPEN WEB #################### \n")
    print("\nBeginning training tokenizer on Open Web Dataset...\n")
    start_time = time.time()

    owt_vocabulary, owt_merges = train_bpe_tokenizer(
        "data/owt_train.txt", 
        vocab_size=32000,
        special_tokens=["<|endoftext|>"], 
        multiprocess=False
    )
    end_time = time.time()
    print(f"\nTraining BPE tokenizer on Open Web dataset took {format_time_duration(end_time - start_time)}\n")

    print(f"\nSaving to Disk...\n")
    save_start_time = time.time()
    os.makedirs("artifacts/tokenizer/owt", exist_ok=True)
    with open("artifacts/tokenizer/owt/vocab.pkl", "wb") as f: 
        pickle.dump(owt_vocabulary, f)
    
    with open("artifacts/tokenizer/owt/merges.pkl", "wb") as f: 
        pickle.dump(owt_merges, f)
    save_end_time = time.time()
    print(f"\n Saving to disk took {format_time_duration(save_end_time - save_start_time)}\n")

    size_in_bytes = get_folder_size_pathlib("artifacts/tokenizer/owt")
    print(f"Memory usage of tokenizer on Open Web training: {format_byte_size(size_in_bytes)}")
    