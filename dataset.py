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
    """准备数据集"""
    print("加载数据集...")
    dataset = load_dataset("t1annnnn/Chinese_sentimentAnalyze")

    # 使用预分割的数据集
    train_data = dataset["train"].shuffle(seed=42)
    test_data = dataset["test"].shuffle(seed=42)

    # 限制数据量
    train_data = train_data.select(range(min(config.TRAIN_SIZE, len(train_data))))
    test_data = test_data.select(range(min(config.TEST_SIZE, len(test_data))))

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

    # 导出 JSONL
    import json
    with open("data/train_data.jsonl", "w", encoding="utf-8") as f:
        for text, label in zip(train_texts, train_labels):
            f.write(json.dumps({
                "text": text,
                "label": label,
                "sentiment": "积极" if label == 1 else "消极"
            }, ensure_ascii=False) + "\n")

    with open("data/test_data.jsonl", "w", encoding="utf-8") as f:
        for text, label in zip(test_texts, test_labels):
            f.write(json.dumps({
                "text": text,
                "label": label,
                "sentiment": "积极" if label == 1 else "消极"
            }, ensure_ascii=False) + "\n")

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print("数据已保存到 data/data.pt, vocab.json, train_data.jsonl, test_data.jsonl")

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
