"""导出数据集为 JSONL 格式"""

import torch
import json
import config


def export_jsonl():
    data = torch.load(f"data/data_{config.DATA_SIZE_NAME}.pt", weights_only=False)

    # 导出训练集
    with open(f"data/train_data_{config.DATA_SIZE_NAME}.jsonl", "w", encoding="utf-8") as f:
        for text, label in zip(data["train_texts"], data["train_labels"]):
            f.write(json.dumps({
                "text": text,
                "label": label,
                "sentiment": "积极" if label == 1 else "消极"
            }, ensure_ascii=False) + "\n")

    # 导出测试集
    with open(f"data/test_data_{config.DATA_SIZE_NAME}.jsonl", "w", encoding="utf-8") as f:
        for text, label in zip(data["test_texts"], data["test_labels"]):
            f.write(json.dumps({
                "text": text,
                "label": label,
                "sentiment": "积极" if label == 1 else "消极"
            }, ensure_ascii=False) + "\n")

    print(f"训练集已导出: train_data_{config.DATA_SIZE_NAME}.jsonl ({len(data['train_texts'])} 条)")
    print(f"测试集已导出: test_data_{config.DATA_SIZE_NAME}.jsonl ({len(data['test_texts'])} 条)")


if __name__ == "__main__":
    export_jsonl()
