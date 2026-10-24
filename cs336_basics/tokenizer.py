import regex as re
import heapq
from collections.abc import Iterable, Iterator
import pickle

class Element:
    def __init__(self, value, next_, prev):
        self.value = value
        self.next = next_
        self.prev = prev
        self.is_deleted = False

class MinMap:
    def __init__(self):
        self.d = {}
        self.heap = []

    def put(self, key, value):
        self.d[key] = value
        heapq.heappush(self.heap, (value, key))

    def delete(self, key):
        self.d.pop(key, None)

    def get(self, key):
        return self.d[key]

    def contain(self, key):
        return key in self.d

    def get_min_key(self):
        while self.heap:
            value, key = self.heap[0]
            if key in self.d and self.d[key] == value:
                return key
            heapq.heappop(self.heap)
        return None

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges

        self.vocab_map = {}
        for key in vocab:
            self.vocab_map[vocab[key]] = key

        self.merges_map = {}
        for i in range(len(merges)):
            self.merges_map[merges[i]] = i

        self.all_special_token_map = {}
        self.all_special_token_rev_map = {}

        for i in range(len(merges) + 256, len(vocab)):
            self.all_special_token_map[vocab[i]] = i
            self.all_special_token_rev_map[i] = vocab[i]

        if special_tokens:
            iterator_index = len(vocab)
            for special_token in special_tokens:
                token_utf8 = special_token.encode("utf-8")
                if token_utf8 in self.all_special_token_map:
                    continue
                self.all_special_token_map[token_utf8] = iterator_index
                self.all_special_token_rev_map[iterator_index] = token_utf8
                iterator_index += 1
    def from_files(vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text:str) -> list[int]:
        pretoken_list, special_tokens_index_set = self._split_chunk(text)
        token_list = []
        for i in range(len(pretoken_list)):
            chunk = pretoken_list[i]
            if i in special_tokens_index_set:
                token_list.append(self._encode_special_token(chunk))
                continue
            token_list.extend(self._encode_chunk(chunk))
        return token_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids:list[int]) -> str:
        byte_list = []
        for token in ids:
            if token in self.all_special_token_rev_map:
                byte_llist.append(self.all_special_token_rev_map[token])
                continue
            byte_list.append(self.vocab[token])
        return b"".join(byte_list).decode("utf-8", errors="ignore")

    def _split_chunk(self, text):

        special_tokens = [tok for tok in self.all_special_token_map]
        special_tokens.sort(key=len, reverse=True)

        split_pat = "|".join(re.escape(tok.decode("utf-8", errors="ignore")) for tok in special_tokens)
        chunks = re.split(f"({split_pat})", text)

        special_tokens_index_set = set()
        pretoken_list = []

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        id_odd = False

        for chunk in chunks:
            if is_odd:
                pretoken_list.append(chunk)
                special_tokens_index_set.add(len(pretoken_list-1))
            else:
                pretoken_list.extend(re.findall(PAT, chunk))
        return pretoken_list, special_tokens_index_set

    def _encode_special_token(self, text):
        return self.all_special_token_map[text.encode("utf-8")]

    def _encode_chunk(self, pretoken):
        pretoken_byte_rep = []
        pretoken_utf8 = pretoken.encode("utf-8")
        for i in range(len(pretoken_utf8)):
            next_ = -1 if i==len(pretoken_utf8)-1 else i+1
            prev = i-1
            pretoken_byte_rep.append(Element(bytes([pretoken_utf8[i]]), next_, prev))

        # init
        min_map = MinMap()

        for i in range(len(pretoken_byte_rep)-1):
            b1 = pretoken_byte_rep[i].value
            b2 = pretoken_byte_rep[i+1].value
            byte_pair = (b1, b2)
            if byte_pair in self.merges_map:
                min_map.put(i, self.merges_map[byte_pair])

        while True:
            key = min_map.get_min_key()
            if key is None:
                break
            min_map.delete(key)
            byte_index = key
            element = pretoken_byte_rep[byte_index]

            if element.is_deleted:
                continue

            prev_index = element.prev
            next_index = element.next
            next_element = pretoken_byte_rep[next_index]
            next2_index = next_element.next

            element.value = element.value + next_element.value
            element.next = next2_index
            next_element.is_deleted = True

            if prev_index != -1:
                prev_element = pretoken_byte_rep[prev_index]
                min_map.delete(prev_index)
                new_byte_pair = (prev_element.value, element.value)
                if new_byte_pair in self.merges_map:
                    min_map.put(prev_index, self.merges_map(new_byte_pair))

            if next2_index != -1:
                next2_element = pretoken_byte_rep[next2_index]
                min_map.delete(next_index)
                new_byte_pair = (element.value, next2_element.value)
                if new_byte_pair in self.merges_map:
                    min_map.put(byte_index, self.merges_map[new_byte_pair])
                next2_element.prev = byte_index

        final_rep = []
        index = 0
        while index != -1:
            final_rep.append(self.vocab_map[pretoken_byte_rep[index]].value)
            indenx = pretoken_byte_rep[index].next

        return final_rep
