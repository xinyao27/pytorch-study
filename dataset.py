"""数据集准备和加载"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from tokenizer import SimpleTokenizer
import config


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx], self.max_length)
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def prepare_data():
    """准备 IMDB 数据集"""
    print("加载 IMDB 数据集...")
    dataset = load_dataset("imdb")

    # 打乱数据确保标签分布均匀
    train_data = dataset["train"].shuffle(seed=42).select(range(config.TRAIN_SIZE))
    test_data = dataset["test"].shuffle(seed=42).select(range(config.TEST_SIZE))

    train_texts = train_data["text"]
    train_labels = train_data["label"]
    test_texts = test_data["text"]
    test_labels = test_data["label"]

    print(f"标签分布 - 训练集: 积极={sum(train_labels)}, 消极={len(train_labels)-sum(train_labels)}")

    # 构建词表
    print("构建词表...")
    tokenizer = SimpleTokenizer(vocab_size=config.VOCAB_SIZE)
    tokenizer.build_vocab(train_texts)
    tokenizer.save("data/vocab.json")

    # 创建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.MAX_LENGTH)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, config.MAX_LENGTH)

    # 保存处理后的数据
    torch.save({
        "train_texts": train_texts,
        "train_labels": train_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
    }, "data/data.pt")

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print("数据已保存到 data.pt 和 vocab.json")

    return train_dataset, test_dataset, tokenizer


def load_data():
    """加载已处理的数据"""
    data = torch.load("data/data.pt", weights_only=False)
    tokenizer = SimpleTokenizer(vocab_size=config.VOCAB_SIZE)
    tokenizer.load("data/vocab.json")

    train_dataset = SentimentDataset(
        data["train_texts"], data["train_labels"], tokenizer, config.MAX_LENGTH
    )
    test_dataset = SentimentDataset(
        data["test_texts"], data["test_labels"], tokenizer, config.MAX_LENGTH
    )

    return train_dataset, test_dataset, tokenizer


def get_dataloaders(train_dataset, test_dataset):
    """获取 DataLoader"""
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    return train_loader, test_loader


if __name__ == "__main__":
    prepare_data()
