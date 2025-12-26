"""简易分词器"""

from collections import Counter


class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<CLS>"}

    def build_vocab(self, texts):
        """从文本构建词表"""
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        most_common = word_counts.most_common(self.vocab_size - len(self.word2idx))
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"词表大小: {len(self.word2idx)}")

    def encode(self, text, max_length=128):
        """将文本编码为 token ids"""
        words = text.lower().split()
        tokens = [self.word2idx.get("<CLS>")]
        for word in words[:max_length - 1]:
            tokens.append(self.word2idx.get(word, self.word2idx["<UNK>"]))

        if len(tokens) < max_length:
            tokens += [self.word2idx["<PAD>"]] * (max_length - len(tokens))

        return tokens[:max_length]

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
