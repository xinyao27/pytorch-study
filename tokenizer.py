"""简易分词器 - 支持中文字符级分词"""

import re
from collections import Counter


def tokenize_chinese(text):
    """中文字符级分词，保留英文单词和数字"""
    tokens = []
    # 匹配: 中文字符 | 英文单词 | 数字 | 微博表情[xxx]
    pattern = r'\[[\u4e00-\u9fff\w]+\]|[\u4e00-\u9fff]|[a-zA-Z]+|[0-9]+'
    tokens = re.findall(pattern, text)
    return tokens


class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<CLS>"}

    def build_vocab(self, texts):
        """从文本构建词表"""
        word_counts = Counter()
        for text in texts:
            tokens = tokenize_chinese(text)
            word_counts.update(tokens)

        most_common = word_counts.most_common(self.vocab_size - len(self.word2idx))
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"词表大小: {len(self.word2idx)}")

    def encode(self, text, max_length=128):
        """将文本编码为 token ids"""
        tokens = tokenize_chinese(text)
        ids = [self.word2idx.get("<CLS>")]
        for token in tokens[:max_length - 1]:
            ids.append(self.word2idx.get(token, self.word2idx["<UNK>"]))

        if len(ids) < max_length:
            ids += [self.word2idx["<PAD>"]] * (max_length - len(ids))

        return ids[:max_length]

    def decode(self, tokens):
        """将 token ids 解码为文本"""
        return " ".join([self.idx2word.get(t, "<UNK>") for t in tokens if t != 0])

    def save(self, path):
        """保存词表"""
        import json
        with open(path, "w") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)

    def load(self, path):
        """加载词表"""
        import json
        with open(path, "r") as f:
            self.word2idx = json.load(f)
            self.idx2word = {v: k for k, v in self.word2idx.items()}
