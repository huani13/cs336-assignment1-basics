import regex as re
import time
from multiprocessing import Pool
import os
from typing import BinaryIO

class Element:
    def __init__(self, value, next_, prev):
        self.value = value
        self.next = next_
        self.prev = prev
        self.is_deleted = False

class MaxMap:
    def __init__(self):
        self.d = {}
        self.heap = []
        self.pos = {}

    def _higher(self, a, b):
        return (self.d[a], a) > (self.d[b], b)

    def _swap(self, i, j):
        heap = self.heap
        heap[i], heap[j] = heap[j], heap[i]
        self.pos[heap[i]] = i
        self.pos[heap[j]] = j

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if not self._higher(self.heap[i], self.heap[parent]):
                break
            self._swap(i, parent)
            i = parent

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            best = i
            left = 2 * i + 1
            right = left + 1
            if left < n and self._higher(self.heap[left], self.heap[best]):
                best = left
            if right < n and self._higher(self.heap[right], self.heap[best]):
                best = right
            if best == i:
                break
            self._swap(i, best)
            i = best

    def put(self, key, value):
        if key in self.pos:
            old_value = self.d[key]
            self.d[key] = value
            if value > old_value:
                self._sift_up(self.pos[key])
            elif value < old_value:
                self._sift_down(self.pos[key])
        else:
            self.d[key] = value
            self.heap.append(key)
            self.pos[key] = len(self.heap) - 1
            self._sift_up(len(self.heap) - 1)

    def delete(self, key):
        if key not in self.pos:
            return
        i = self.pos.pop(key)
        del self.d[key]
        last = self.heap.pop()
        if i < len(self.heap):
            self.heap[i] = last
            self.pos[last] = i
            self._sift_up(i)
            self._sift_down(self.pos[last])

    def get(self, key):
        return self.d[key]

    def get_max_key(self):
        return self.heap[0] if self.heap else None

def get_pretoken_count(text, special_tokens):
    split_pat = "|".join(re.escape(tok) for tok in special_tokens)
    chunks = re.split(split_pat, text)
    pretoken_list = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for chunk in chunks:
        pretoken_list.extend(re.findall(PAT, chunk))

    pretoken_count_map = {}
    for pretoken in pretoken_list:
        if pretoken not in pretoken_count_map:
            pretoken_count_map[pretoken] = 0
        pretoken_count_map[pretoken] += 1

    return pretoken_count_map

def get_pretoken_count_wrapper(start, end, filename, special_tokens):
    t0 = time.perf_counter()
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    resp = get_pretoken_count(chunk, special_tokens)
    return resp, time.perf_counter() - t0, os.getpid()

def aggregate_pretoken_count(l):
    total_pretoken_count = {}
    for pretoken_count_map in l:
        for pretoken in pretoken_count_map:
            if pretoken not in total_pretoken_count:
                total_pretoken_count[pretoken] = 0
            total_pretoken_count[pretoken] += pretoken_count_map[pretoken]
    return total_pretoken_count

def bpe_logic(pretoken_count_map, vocab_size, special_tokens):
    pretoken_index_list = []
    pretoken_index_map = {}
    i = 0
    for pretoken in pretoken_count_map:
        pretoken_index_list.append(pretoken)
        pretoken_index_map[pretoken] = i
        i += 1
    pretoken_byte_rep_list = [[] for i in range(len(pretoken_index_list))]
    for pretoken_index in range(len(pretoken_index_list)):
        pretoken = pretoken_index_list[pretoken_index]
        pretoken_utf8 = pretoken.encode("utf-8")
        for i in range(len(pretoken_utf8)):
            next_ = -1 if i == len(pretoken_utf8)-1 else i+1
            prev = i-1
            pretoken_byte_rep_list[pretoken_index].append(Element(bytes([pretoken_utf8[i]]), next_, prev))

    vocab_length = vocab_size
    initial_vocab_length = 256

    count_merge = MaxMap()
    count_merge_ref = {}

    # pre count
    for pretoken_index in range(len(pretoken_index_list)):
        pretoken_count = pretoken_count_map[pretoken_index_list[pretoken_index]]
        for i in range(len(pretoken_byte_rep_list[pretoken_index])-1):
            b1 = pretoken_byte_rep_list[pretoken_index][i].value
            b2 = pretoken_byte_rep_list[pretoken_index][i+1].value
            key = (b1, b2)
            if key not in count_merge_ref:
                count_merge.put(key, 0)
                count_merge_ref[key] = set()
            count_merge.put(key, count_merge.get(key) + pretoken_count)
            count_merge_ref[key].add((pretoken_index, i))

    vocab = {}
    # build base vocab
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[vocab_length-len(special_tokens)+i] = special_tokens[i].encode("utf-8")
    merges = []

    for i in range(initial_vocab_length, vocab_length-len(special_tokens)): # iterate number of merges
        bytes_to_merge = count_merge.get_max_key()
        merged_byte = bytes_to_merge[0] + bytes_to_merge[1]
        vocab[i] = merged_byte
        merges.append((bytes(bytes_to_merge[0]), bytes(bytes_to_merge[1])))

        # update
        for reference in sorted(count_merge_ref[bytes_to_merge]):
            pretoken_index, byte_index = reference
            pretoken_ref = pretoken_byte_rep_list[pretoken_index]
            pretoken_count = pretoken_count_map[pretoken_index_list[pretoken_index]]

            element = pretoken_ref[byte_index]

            if element.is_deleted:
                continue

            prev_index = element.prev
            next_index = element.next
            next_element = pretoken_ref[next_index]
            next2_index = next_element.next

            element.value = merged_byte
            element.next = next2_index
            next_element.is_deleted = True

            # update prev
            if prev_index != -1:
                prev_element = pretoken_ref[prev_index]
                prev_byte_pair = (prev_element.value, bytes_to_merge[0])
                if prev_byte_pair != bytes_to_merge:
                    count_merge.put(prev_byte_pair, count_merge.get(prev_byte_pair) - pretoken_count)
                    count_merge_ref[prev_byte_pair].remove((pretoken_index, prev_index))

                new_byte_pair = (prev_element.value, merged_byte)
                if new_byte_pair not in count_merge_ref:
                    count_merge.put(new_byte_pair, 0)
                    count_merge_ref[new_byte_pair] = set()
                count_merge.put(new_byte_pair, count_merge.get(new_byte_pair) + pretoken_count)
                count_merge_ref[new_byte_pair].add((pretoken_index, prev_index))

            # update next
            if next2_index != -1:
                next2_element = pretoken_ref[next2_index]
                next_bytes_pair = (next_element.value, next2_element.value)
                if (next_bytes_pair != bytes_to_merge):
                    count_merge.put(next_bytes_pair, count_merge.get(next_bytes_pair) - pretoken_count)
                    count_merge_ref[next_bytes_pair].remove((pretoken_index, next_index))

                next2_element.prev = byte_index

                new_byte_pair = (merged_byte, next2_element.value)
                if new_byte_pair not in count_merge_ref:
                    count_merge.put(new_byte_pair, 0)
                    count_merge_ref[new_byte_pair] = set()
                count_merge.put(new_byte_pair, count_merge.get(new_byte_pair) + pretoken_count)
                count_merge_ref[new_byte_pair].add((pretoken_index, byte_index))

        count_merge.delete(bytes_to_merge)
        del count_merge_ref[bytes_to_merge]

    return vocab, merges

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

def execute_bpe_train(input_path, vocab_size, special_tokens, num_processes=None, num_chunks=None):
    if num_processes is None:
        num_processes = os.cpu_count()
    if num_chunks is None:
        num_chunks = num_processes * 4

    total_start = time.perf_counter()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    args = [(s, e, input_path, special_tokens) for s,e in zip(boundaries[:-1], boundaries[1:])]

    with Pool(num_processes) as pool:
        results = pool.starmap(get_pretoken_count_wrapper, args)

    # for chunk_index, ((s, e, _, _), (_, elapsed, pid)) in enumerate(zip(args, results)):
    #     print(f"[pretoken] chunk {chunk_index} (pid {pid}, {(e - s) / 1e6:.2f} MB): {elapsed:.2f}s")

    final_pretoken_count_map = aggregate_pretoken_count(r[0] for r in results)
    pretoken_end = time.perf_counter()
    # print(f"[pretoken] finished in {pretoken_end - total_start:.2f}s ({len(final_pretoken_count_map)} unique pretokens)")

    vocab, merges = bpe_logic(final_pretoken_count_map, vocab_size, special_tokens)
    total_end = time.perf_counter()
    # print(f"[merge] finished in {total_end - pretoken_end:.2f}s")
    # print(f"[total] finished in {total_end - total_start:.2f}s")

    return vocab, merges
