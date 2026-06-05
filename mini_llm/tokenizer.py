import regex as re
import heapq
from collections import defaultdict
import multiprocessing
from dataclasses import dataclass
from typing import BinaryIO
import os
from typing import Iterable, Iterator, cast
import json
from tests.common import gpt2_bytes_to_unicode

input_file = 'data/small_tiny.txt'
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
vocab_size = 600


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

def count_word_frequencies_parallel(input_file, special_tokens: list[str] | None = None):
    """计算输入文件 input_file 中每个词语的频率"""
    # 1. 获取区块边界
    num_processes = 4
    with open(input_file, "rb") as f:
        
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        tasks = [(start, end, input_file, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    # 2. 创建进程池并分配任务
    with multiprocessing.Pool(processes=num_processes) as pool:
        # starmap 用于处理有多个参数的worker函数
        list_of_dicts = pool.starmap(process_chunk, tasks)
        
    # 3. 合并所有结果
    final_counts = defaultdict(int)
    for local_dict in list_of_dicts:
        # ... 把 local_dict 合并到 final_counts 里 ...
        for k, v in local_dict.items():
            final_counts[k] += v
            
    return final_counts
    

def process_chunk(start, end, input_file, special_tokens: list[str] | None = None):
    """统计一个chunk中的词频"""
    # 打开文件
    chunk_word_counts = defaultdict(int)
    with open(input_file, 'rb') as f:
        
    # 预分词
        f.seek(start)
        chunk = f.read(end-start).decode("utf-8", errors='ignore')
    #     pre_tokens = re.findall(PAT, chunk)

    # # 统计词频
    #     for token in pre_tokens:
    #         token_bytes = token.encode('utf-8')
    #         chunk_word_counts[tuple(token_bytes)] += 1
    # 使用迭代器, 节省内存
        if special_tokens:
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(tok) for tok in sorted_tokens]
            pattern = '(' + "|".join(escaped_tokens) + ')' # 括号代表保留special token
            parts = re.split(pattern, chunk)
        else:
            parts = [chunk]

        for i, part in enumerate(parts):
            if special_tokens and i % 2 == 1:
                # special token本身不参与BPE统计，避免被学习成子片段
                continue
            for match in re.finditer(PAT, part): # 迭代器
                word_bytes = tuple(match.group(0).encode("utf-8"))
                chunk_word_counts[word_bytes] += 1
    # 返回词频
    return chunk_word_counts

def iter_adjacent_pairs(word: tuple[int, ...]) -> Iterator[tuple[int, int]]:
    for a, b in zip(word[:-1], word[1:]):
        yield (a, b)


def count_pairs_in_word(word: tuple[int, ...]) -> dict[tuple[int, int], int]:
    """统计一个word内部每个pair出现次数（同一pair可能重复出现）"""
    pair_count = defaultdict(int)
    for pair in iter_adjacent_pairs(word):
        pair_count[pair] += 1
    return dict(pair_count)


def build_pair_statistics(word_freqs: dict[tuple[int, ...], int]):
    """
    基于词频构建两类全局结构:
    1) pair_freq: 每个pair在全语料中的频次
    2) pair_to_words: 倒排索引，记录每个pair出现在哪些word里
    """
    pair_freq = defaultdict(int)
    pair_to_words = defaultdict(set)

    for word, freq in word_freqs.items():
        local_pair_count = count_pairs_in_word(word)
        for k, v in local_pair_count.items():
            # 1.更新全局频次
            pair_freq[k] += v * freq
            
            # 2. 更新倒排索引
            pair_to_words[k].add(word)
        
    return pair_freq, pair_to_words


@dataclass
class PairHeapItem:
    neg_freq: int
    left_bytes: bytes
    right_bytes: bytes
    pair: tuple[int, int]

    def __lt__(self, other: "PairHeapItem") -> bool:
        # 频率更高优先（neg_freq更小）
        if self.neg_freq != other.neg_freq:
            return self.neg_freq < other.neg_freq
        # tie-break: 左token bytes更大优先
        if self.left_bytes != other.left_bytes:
            return self.left_bytes > other.left_bytes
        # tie-break: 右token bytes更大优先
        if self.right_bytes != other.right_bytes:
            return self.right_bytes > other.right_bytes
        # 最后再按pair id保证稳定性
        return self.pair > other.pair


def push_pair_to_pq(
    pq: list[PairHeapItem],
    pair: tuple[int, int],
    freq: int,
    get_token_bytes,
):
    # 记一下，python的heapq默认是最小堆，需要最高freq的话，就要把key设置成 -freq
    if freq <= 0:
        return
    heapq.heappush(
        pq,
        PairHeapItem(
            neg_freq=-freq,
            left_bytes=get_token_bytes(pair[0]),
            right_bytes=get_token_bytes(pair[1]),
            pair=pair,
        ),
    )


def pop_valid_best_pair(pq: list[PairHeapItem], pair_freq: dict[tuple[int, int], int]):
    """
    懒惰删除策略:
    - PQ里可能有陈旧项
    - 弹出后与pair_freq中的当前频率比对，不一致就跳过
    """
    while pq:
        item = heapq.heappop(pq)
        pair = item.pair
        cur_freq = pair_freq.get(pair, 0)
        if cur_freq <= 0 or cur_freq != -item.neg_freq:
            continue
        return pair, cur_freq
    return None, 0

def merged_pair_in_word(word: tuple[int,...], pair: tuple[int, int], token_cnt: int):
    """在一个 token_id 序列中，将所有出现的指定字节对替换为新的token ID。"""
    merged = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
            merged.append(token_cnt)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)

def flatten_vocab(vocab: dict[int, bytes | tuple[int, int]]) -> dict[int, bytes]:
    cache = {}
    
    def get_token_bytes(tid: int) -> bytes:
        val = vocab[tid]
        if isinstance(val, bytes):
            cache[tid] = val
            return val
        else:
            left_id, right_id = cast(tuple[int, int], val)
            cache[tid] = get_token_bytes(left_id) + get_token_bytes(right_id)
            return cache[tid]
    
    return {k: get_token_bytes(k) for k, v in vocab.items()}


def train_bpe(input_file: str, 
              vocab_size: int,
              special_tokens: list[str] | None = None
              ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # ======= 1. 初始化 =======
    # 1) 词表与merge列表
    vocab: dict[int, bytes | tuple[int, int]] = {i: bytes([i]) for i in range(256)}
    token_cnt = 256
    merges = []

    unique_special_tokens: list[bytes] = []
    if special_tokens:
        seen_special = set()
        for tok in special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in seen_special:
                seen_special.add(tok_bytes)
                unique_special_tokens.append(tok_bytes)

    # 需要给special token预留词表位置
    target_vocab_size = vocab_size - len(unique_special_tokens)

    # 2) 统计初始word频率（word由token_id tuple表示）
    # 需要注意的是，我们合并的 byte pair 一定是word内部的pair， 所以以后就只用遍历 word_freq 啦！
    word_freqs = count_word_frequencies_parallel(input_file, special_tokens=special_tokens)

    # 3) token_id -> bytes 的递归展开（用于记录merges和PQ tie-break）
    token_bytes_cache: dict[int, bytes] = {}

    def get_token_bytes(tid: int) -> bytes:
        if tid in token_bytes_cache:
            return token_bytes_cache[tid]
        val = vocab[tid]
        if isinstance(val, bytes): # 如果是初始的老资历token
            token_bytes_cache[tid] = val
        else:
            # 如果是token id pair
            left_id, right_id = cast(tuple[int, int], val)
            token_bytes_cache[tid] = get_token_bytes(left_id) + get_token_bytes(right_id)
        return token_bytes_cache[tid]

    # 4) 构建全局pair统计 + 倒排索引
    pair_freq, pair_to_words = build_pair_statistics(word_freqs)

    # 5) 初始化优先队列（最大堆通过neg_freq + bytes tie-break实现）
    pq: list[PairHeapItem] = []
    for pair, freq in pair_freq.items():
        push_pair_to_pq(pq, pair, freq, get_token_bytes)

    # ======= 2. 训练循环（增量更新） =======
    while token_cnt < target_vocab_size:
        # 1) 选出当前最优pair（自动跳过陈旧PQ项）
        best_pair, best_freq = pop_valid_best_pair(pq, pair_freq)
        if best_pair is None or best_freq <= 0:
            break

        # 2) 创建新token，并记录byte级merge
        new_token_id = token_cnt
        token_cnt += 1
        vocab[new_token_id] = best_pair
        merges.append((get_token_bytes(best_pair[0]), get_token_bytes(best_pair[1])))

        # 3) 只处理“包含best_pair”的受影响word（倒排索引的价值）
        affected_words = list(pair_to_words.get(best_pair, set()))
        if not affected_words:
            continue

        # 4) 从旧word撤销贡献，并收集新word增量
        # delta_new_words[new_word] = 该new_word新增的总频次
        delta_new_words = defaultdict(int)
        changed_pairs = set()

        for old_word in affected_words:
            old_freq = word_freqs.get(old_word, 0)
            if old_freq <= 0:
                continue

            new_word = merged_pair_in_word(old_word, best_pair, new_token_id)
            if new_word == old_word:
                continue

            # 4a) old_word对pair统计的贡献撤销
            old_local_pair_count = count_pairs_in_word(old_word)
            for pair, cnt in old_local_pair_count.items():
                pair_freq[pair] -= cnt * old_freq
                changed_pairs.add(pair)

                bucket = pair_to_words.get(pair)
                if bucket is not None:
                    bucket.discard(old_word)
                    if not bucket:
                        pair_to_words.pop(pair, None)

            # 4b) old_word从当前语料状态中移除
            word_freqs.pop(old_word, None)
            delta_new_words[new_word] += old_freq

        # 5) 把new_word贡献加回全局结构
        for new_word, add_freq in delta_new_words.items():
            word_freqs[new_word] += add_freq
            new_local_pair_count = count_pairs_in_word(new_word)
            for pair, cnt in new_local_pair_count.items():
                # 懒更新体现之处：
                # 我们修改全局结构 pair_freq的时候不去同步修改pq
                # 因为堆不支持随机修改
                # 而是弹出时检测！
                pair_freq[pair] += cnt * add_freq
                pair_to_words[pair].add(new_word)
                changed_pairs.add(pair)

        for pair in changed_pairs:
            push_pair_to_pq(pq, pair, pair_freq.get(pair, 0), get_token_bytes)
        

    
    final_vocab = flatten_vocab(vocab)
    
    if special_tokens:
        vocab_size = len(final_vocab)
        for tok_bytes in unique_special_tokens:
            final_vocab[vocab_size] = tok_bytes
            vocab_size += 1
    return final_vocab, merges

def find_merge_in_word(word: list[int], merge: tuple):
    for a, b in zip(word[:-1], word[1:]):
        if (a, b) == merge:
            return True
    return False

class Tokenizer():
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list, 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        vocab_size = len(vocab) - 1
        if special_tokens:
            for token in special_tokens:
                self.vocab[vocab_size] = token.encode('utf-8')
                vocab_size += 1
        self.vocab_size = vocab_size
        self.encode_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}
    
    def find_best_merge(self, word: list[int]):
        pairs = []
        for i in range(len(word) - 1):
            # 获取 word[i] 对应的 bytes
            left_bytes = self.vocab[word[i]]
            right_bytes = self.vocab[word[i + 1]]
            pair = (left_bytes, right_bytes)
            if pair in self.merge_rank:
                pairs.append((self.merge_rank[pair], pair))
        
        if not pairs:
            return None
        
        # 返回优先级最高的合并规则
        return min(pairs)[1]
    
    def encode(self, text: str) -> list[int]:
        final_token_ids = []
        
        if self.special_tokens:
            # import pdb; pdb.set_trace()
            # 1. 转义
            # key: 排序的键, reverse: 降序
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(tok) for tok in sorted_tokens]

            # 2. 拼成整个正则表达式
            # 这里加括号的话，内部字符串就会被 split保留
            pattern = '(' + "|".join(escaped_tokens) + ')'
            
            # 3. 分割
            words = re.split(pattern, text)
        else:
            words = [text]
            
        # 2. 遍历 words, 在每个word内部查看是否有可用的合并, 如果有, 合并之
        # 不变量: words 始终是token id序列
        for i, word in enumerate(words):
            if self.special_tokens and i % 2 == 1:
                token_id = self.encode_vocab[word.encode('utf-8')]
                final_token_ids.append(token_id)
            
            else:
                # word转换为字节的列表
                # token_ids = list(word.encode('utf-8'))
                # 这里是gpt2的预分词
                for match in re.finditer(PAT, word):
                    token_str = match.group(0)
                    byte_vals = list(token_str.encode('utf-8'))
                    token_ids = [self.encode_vocab[bytes([b])] for b in byte_vals]
                    while len(token_ids) > 1:
                        merge = self.find_best_merge(token_ids)
                        # print(f"find merge: {merge}")
                        if merge:
                            merged_bytes = merge[0] + merge[1]
                            new_token_id = self.encode_vocab[merged_bytes]
                            merge_ints = (self.encode_vocab[merge[0]], self.encode_vocab[merge[1]])
                            token_ids = list(merged_pair_in_word(tuple(token_ids), merge_ints, new_token_id))
                        else:
                            break
                    
                    final_token_ids.extend(token_ids)
        # 3. 返回
        return final_token_ids
    
    
        
    def decode(self, input_ids: list[int]) -> str:
        # 1. token_id -> 字节对
        byte_pairs = [self.vocab[token_id] for token_id in input_ids]
        
        # 2. 拼接
        byte_chunk = b''.join(byte_pairs)
        
        # 3. 解码
        return byte_chunk.decode("utf-8", errors="replace")

    @classmethod 
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 加载 GPT-2 的字节映射表
        def gpt2_bytes_to_unicode():
            """
            GPT-2 使用的字节到 Unicode 映射
            """
            bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
            cs = bs[:]
            n = 0
            for b in range(2**8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2**8+n)
                    n += 1
            cs = [chr(n) for n in cs]
            return dict(zip(bs, cs))
        
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        
        # 加载 vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            gpt2_vocab = json.load(f)
        
        vocab = {}
        for token_str, token_id in gpt2_vocab.items():
            vocab[token_id] = bytes([gpt2_byte_decoder[char] for char in token_str])
        
        # 加载 merges（跳过注释与非法行）
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过注释行
                    continue
                parts = line.split(' ')
                if len(parts) != 2:  # 确保是有效的两列
                    continue
                a, b = parts
                a_bytes = bytes([gpt2_byte_decoder[char] for char in a])
                b_bytes = bytes([gpt2_byte_decoder[char] for char in b])
                merges.append((a_bytes, b_bytes))
        
        return cls(vocab, merges, special_tokens)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
if __name__=="__main__":
    special_tokens = ['<ttt>', '[ffff]']
    vocab, merges = train_bpe(input_file, vocab_size)
    # print("======vocab======")
    # for k, v in vocab.items():
    #     print(f"{k}: {v}")
    
    print("======merges======")
    for merge in merges:
        print(merge)
    # 构造Tokenizer
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    new_token_id = tokenizer.encode_vocab['<ttt>'.encode('utf-8')]
    print(f"new token id = {new_token_id}")
    # 测试文本
    test_text = "I love MiniLLM!"

    print(f"test: {test_text}")
    
    # 编码
    encoded = tokenizer.encode(test_text)
    print("Encoded:", encoded)

    # 解码
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
    


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return Tokenizer(vocab, merges, special_tokens)
