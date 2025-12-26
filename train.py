"""训练脚本"""

import torch
import torch.nn as nn
from tqdm import tqdm

from model import SentimentLLM
from dataset import load_data, get_dataloaders
import config


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    device = get_device()
    print(f"使用设备: {device}")

    # 加载数据
    print("\n加载数据...")
    train_dataset, test_dataset, tokenizer = load_data()
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset)

    # 创建模型
    print("\n创建模型...")
    model = SentimentLLM(
        vocab_size=len(tokenizer.word2idx),
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_ff=config.D_FF,
        max_len=config.MAX_LENGTH,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss()

    # 训练
    print("\n开始训练...")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "models/sentiment_model.pt")
    print("\n模型已保存到 models/sentiment_model.pt")


if __name__ == "__main__":
    main()
