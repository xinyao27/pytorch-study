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
    tokens = torch.tensor([tokenizer.encode(text, config.MAX_LENGTH)]).to(device)

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
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device}")

    model, tokenizer = load_model()
    model = model.to(device)

    test_sentences = [
        "This movie is absolutely wonderful and amazing!",
        "Terrible film, waste of time and money.",
        "It was okay, nothing special.",
        "I love this product, it works great!",
        "Very disappointing experience, would not recommend.",
    ]

    print("\n情感分析结果:")
    print("-" * 60)
    for sentence in test_sentences:
        result = predict(sentence, model, tokenizer, device)
        emoji = "✓" if result["sentiment"] == "积极" else "✗"
        print(f"\n文本: {sentence}")
        print(f"预测: {result['sentiment']} {emoji}")
        print(f"置信度: {result['confidence']:.2%}")


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
