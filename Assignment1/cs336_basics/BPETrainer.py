import os
import regex
import multiprocessing
from collections import defaultdict
from typing import BinaryIO

class BPETrainer:
    def __init__(self):
        self._GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._COMPILED_PAT = regex.compile(self._GPT2_PAT)

    def _process_chunk(self,
        input_path: str | os.PathLike,
        start: int,
        end: int,
        special_tokens: list[str],
    ) -> list[bytes]:
        with open(input_path, 'rb') as fb:
            fb.seek(start)
            chunk = fb.read(end - start).decode('utf-8', errors = 'ignore')
        pattern = "|".join(map(regex.escape, special_tokens))
        sub_chunks = regex.split(pattern, chunk)    # Use special_tokens for splitting 这里将special_token都去掉了
        tokens: list[bytes] = list()
        for sub_chunk in sub_chunks:
            tokens.extend([match.group(0).encode("utf-8") for match in self._COMPILED_PAT.finditer(sub_chunk)])
        return tokens


    def _get_pair_counts(self,
        all_token: list[bytes]
    ) -> tuple[defaultdict[tuple[int, int], int], defaultdict[tuple[int, int], set[int]], list[list[int]]]:
        
        all_pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        all_pair_index: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        all_token_ids: list[list[int]] = list()

        for num, token in enumerate(all_token):
            # 这里能这么做的原因是，token全部拆成单字符，也就是一个字节，属于0-255之间，正好对应vocab的初始值。 
            # 且在process_chunk中已经将special tokens去掉了，所以可以这样处理
            all_token_ids.append(list(token))       
            for pair in zip(token[:-1], token[1:]):     # pair: tuple[int, int]
                all_pair_counts[pair] += 1
                all_pair_index[pair].add(num)

        return all_pair_counts, all_pair_index, all_token_ids


    def _merge_pair(self,
        max_pair: tuple[int, int],
        token_ids: list[int],
        new_token_id: int
    ) -> list[int]:
        new_token_ids: list[int] = list()
        i:int = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == max_pair:
                i += 2
                new_token_ids.append(new_token_id)
            else:
                new_token_ids.append(token_ids[i])
                i += 1
        return new_token_ids


    # 官方提供
    def _find_chunk_boundaries(self,
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


    def train_bpe(self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 8,
        **kwargs,
    ) -> tuple[defaultdict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
        """
        vocab = {i: bytes([i]) for i in range(256)}   # defaultdict[int, bytes]
        merges = []                                 # list[tuple[bytes, bytes]]

        for sp_tok in special_tokens:
            vocab[len(vocab)] = sp_tok.encode("utf-8")
        
        with open(input_path, 'rb') as fb:
            bounds = self._find_chunk_boundaries(fb, num_processes, "<|endoftext|>".encode("utf-8")) # list[start_idx:int]
        tasks = [(input_path, start, end, special_tokens) for start, end in zip(bounds[:-1], bounds[1:])]

        with multiprocessing.Pool(processes = len(tasks)) as pool:
            chunk_results = pool.starmap(self._process_chunk, tasks)  # [[bytes, ...], [bytes, ...], ...]

        all_token: list[bytes] = list()     # [bytes1, bytes2, ...]
        for chunk_result in chunk_results:
            all_token.extend(chunk_result)

        
        all_pair_counts, all_pair_index, all_token_ids = self._get_pair_counts(all_token)    # dict[tuple[int, int], int], dict[tuple[int, int], set[int]], list[list[int]]
        num_merges = vocab_size - len(vocab)

        for i in range(num_merges):
            if not all_pair_counts:
                break 

            '''others'''
            def rank(pair: tuple[int, int]) -> tuple[int, tuple[bytes, bytes]]:
                return all_pair_counts[pair], (vocab[pair[0]], vocab[pair[1]])
            max_pair = max(all_pair_counts, key=rank)

            ## 用这个反而不正确
            # max_pair = max(all_pair_counts, key = all_pair_counts.get)

            new_token = vocab[max_pair[0]] + vocab[max_pair[1]]
            new_token_id = len(vocab)
            vocab[new_token_id] = new_token
            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))

            affected_index = all_pair_index[max_pair].copy()    # all_pair_index 在循环里是要被修改的
            for index in affected_index:
                token_ids = all_token_ids[index]  # token_ids: list[int]
                if len(token_ids) < 2: continue
                for affected_pair in zip(token_ids[:-1], token_ids[1:]):
                    all_pair_counts[affected_pair] -= 1
                    all_pair_index[affected_pair].discard(index)
                    if all_pair_counts[affected_pair] == 0:
                        del all_pair_counts[affected_pair]
                        del all_pair_index[affected_pair]
                
                new_token_ids = self._merge_pair(max_pair, token_ids, new_token_id)
                all_token_ids[index] = new_token_ids
                for increase_pair in zip(new_token_ids[:-1], new_token_ids[1:]):
                    all_pair_counts[increase_pair] += 1
                    all_pair_index[increase_pair].add(index)

        return vocab, merges















