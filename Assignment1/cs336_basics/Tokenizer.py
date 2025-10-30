from typing import Optional
import regex
import base64

class Tokenizer:
    def __init__(self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]],
    ): 
        self.vocab = vocab                                              # int -> bytes
        self.token_to_id = {token: id for id, token in self.vocab.items()}      # bytes -> int
        self.merges = merges
        self.merges_rank = {pair: num for num, pair in enumerate(self.merges)}
        self.special_tokens = sorted(special_tokens or [], key = lambda x: -len(x))       # 长的优先
        self._GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._COMPILED_PAT = regex.compile(self._GPT2_PAT)

    @classmethod
    def from_file(cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: Optional[list[str]],                          
    ):
        # 假设vocab文件格式为: id<\t>base64_encoded_token
        vocab: dict[int, bytes] = dict()
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                id_str, token_str = line.strip().split('\t')
                vocab[int(id_str)] = base64.b64decode(token_str)
        
        # 假设merges文件格式为: token1_str<\t>token2_str
        # 注意merges存的时候，多个符号拼接起来的，它会将多个符号拆开，然后存储，所以不需要base64
        # 但是，单个字节（0–255）不一定能被 UTF-8 正确解码为字符。
        # 所以，实际解决中，我们可以直接映射到 U+0100 到 U+01FF
        def token_str_to_bytes(s: str) -> bytes:
            return bytes(ord(c) - 0x100 for c in s)

        merges: list[tuple[bytes, bytes]] = list()
        with open(merges_path, 'r', encoding="utf-8") as f:
            for line in f:
                part1, part2 = line.strip().split("\t")
                byte1 = token_str_to_bytes(part1)
                byte2 = token_str_to_bytes(part2)
                merges.append((byte1, byte2))
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def _process_text(self,
        input_data: str,
    ) -> list[list[bytes]]:
        
        pattern = "|".join(map(regex.escape, self.special_tokens))
        sub_chunks = regex.split(f"({pattern})", input_data) if pattern else [input_data]   # Use special_tokens for splitting 这里将special_token保留了
        token_bytes: list[list[bytes]] = list()
        for sub_chunk in sub_chunks:
            if sub_chunk in self.special_tokens:
                token_bytes.append([sub_chunk.encode("utf-8")])
            else:
                tokens = [match.group(0).encode("utf-8") for match in self._COMPILED_PAT.finditer(sub_chunk)]
                for token in tokens:
                    token_bytes.append([bytes([b]) for b in token])

        return token_bytes
    
    def _merge(self,
        token_byte: list[bytes],
        new_token: bytes,
        merge_pair: tuple[bytes, bytes],
    ) -> list[bytes]:
        new_token_byte: list[bytes] = list()
        i: int = 0
        while i < len(token_byte):
            if i < len(token_byte) - 1 and (token_byte[i], token_byte[i+1]) == merge_pair:
                new_token_byte.append(new_token)
                i += 2
            else:
                new_token_byte.append(token_byte[i])
                i += 1
        return new_token_byte


    def encode(self,
        input_data: str,           
    ) -> list[int]:
        # 处理 input_data 时要注意，输入的数据可能很大，需要用迭代器处理
        token_ids: list[int] = list()
        token_bytes = self._process_text(input_data)   # token_bytes: list[list[bytes]]
        for token_byte in token_bytes:  # token_byte: list[bytes]
            if len(token_byte) == 1 and token_byte[0] in self.token_to_id:
                token_ids.append(self.token_to_id[token_byte[0]])
                continue
            while len(token_byte) >= 2:
                pairs = list(zip(token_byte[:-1], token_byte[1:]))

                merge_pair = min(pairs, key = lambda x: self.merges_rank.get(x, float('inf')))
                if merge_pair not in self.merges_rank: break
                new_token = merge_pair[0] + merge_pair[1]
                token_byte = self._merge(token_byte, new_token, merge_pair)
            
            token_ids.extend([self.token_to_id[b] for b in token_byte])
        return token_ids
    
    def encode_iterable(self,
        iterable: list[str]
    ) -> iter:
        for line in iterable:
            token_ids = self.encode(line)
            yield from token_ids


    def decode(self,
        ids: list[int],
    ) -> str:
        tokens = b"".join(self.vocab.get(id, "�".encode("utf-8")) for id in ids)
        return tokens.decode(encoding="utf-8", errors='replace')
    