"""推理脚本"""

import torch
from model import SentimentLLM
from tokenizer import SimpleTokenizer
import config


def load_model(model_path="models/sentiment_model.pt", vocab_path="data/vocab.json"):
    """加载模型和分词器"""
    tokenizer = SimpleTokenizer(vocab_size=config.VOCAB_SIZE)
    tokenizer.load(vocab_path)

    model = SentimentLLM(
        vocab_size=len(tokenizer.word2idx),
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_ff=config.D_FF,
        max_len=config.MAX_LENGTH,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model, tokenizer


def predict(text, model, tokenizer, device="cpu"):
    """预测单条文本"""
    encoded = tokenizer.encode(text, config.MAX_LENGTH)
    tokens = torch.tensor([encoded]).to(device)

    # 检查 <UNK> 比例，排除 <PAD> 和 <CLS>
    unk_id = tokenizer.word2idx["<UNK>"]
    pad_id = tokenizer.word2idx["<PAD>"]
    cls_id = tokenizer.word2idx["<CLS>"]
    valid_tokens = [t for t in encoded if t not in (pad_id, cls_id)]
    unk_ratio = sum(1 for t in valid_tokens if t == unk_id) / len(valid_tokens) if valid_tokens else 1

    # 如果大部分是未知词，返回不确定
    if unk_ratio > 0.5:
        return {
            "text": text,
            "sentiment": "不确定",
            "confidence": 0.5,
            "scores": {"消极": 0.5, "积极": 0.5}
        }

    with torch.no_grad():
        logits = model(tokens)
        pred = logits.argmax(dim=-1).item()
        prob = torch.softmax(logits, dim=-1)[0]

    return {
        "text": text,
        "sentiment": "积极" if pred == 1 else "消极",
        "confidence": prob[pred].item(),
        "scores": {"消极": prob[0].item(), "积极": prob[1].item()}
    }


def main():
    import json

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device}")

    model, tokenizer = load_model()
    model = model.to(device)

    # 从测试集读取数据
    test_data = []
    with open("data/test_data.jsonl", "r") as f:
        for line in f:
            test_data.append(json.loads(line))

    # 计算准确率
    correct = 0
    total = len(test_data)

    print(f"\n测试集大小: {total}")
    print("-" * 60)

    # 显示前 10 个样本的详细结果
    for item in test_data[:10]:
        text = item["text"]
        label = item["label"]
        label_name = "积极" if label == 1 else "消极"

        result = predict(text, model, tokenizer, device)
        is_correct = result["sentiment"] == label_name
        emoji = "✓" if is_correct else "✗"

        print(f"\n文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"真实: {label_name} | 预测: {result['sentiment']} {emoji} ({result['confidence']:.1%})")

    # 计算整体准确率
    print("\n" + "-" * 60)
    print("计算整体准确率...")

    for item in test_data:
        result = predict(item["text"], model, tokenizer, device)
        label_name = "积极" if item["label"] == 1 else "消极"
        if result["sentiment"] == label_name:
            correct += 1

    print(f"\n准确率: {correct}/{total} = {correct/total:.2%}")


def interactive():
    """交互式测试"""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device}")
    print("加载模型...")

    model, tokenizer = load_model()
    model = model.to(device)

    print("\n输入文本进行情感分析 (输入 q 退出):")
    print("-" * 40)

    while True:
        text = input("\n> ").strip()
        if text.lower() == "q":
            print("Bye!")
            break
        if not text:
            continue

        result = predict(text, model, tokenizer, device)
        emoji = "✓" if result["sentiment"] == "积极" else "✗"
        print(f"预测: {result['sentiment']} {emoji} ({result['confidence']:.1%})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive()
    else:
        main()
