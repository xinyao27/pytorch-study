"""
训练脚本

本脚本实现了情感分类模型的完整训练流程，包括：
1. 设备选择（CUDA/MPS/CPU）
2. 数据加载与预处理
3. 模型创建与初始化
4. 训练循环与评估
5. 模型保存
"""

import torch  # PyTorch 深度学习框架
import torch.nn as nn  # 神经网络模块
from tqdm import tqdm  # 进度条显示库

from model import SentimentLLM  # 自定义的情感分类模型
from dataset import load_data, get_dataloaders  # 数据加载相关函数
import config  # 配置文件，包含超参数


def get_device():
    """
    自动选择最佳计算设备

    优先级：CUDA GPU > Apple MPS > CPU

    Returns:
        str: 设备名称 ("cuda", "mps", 或 "cpu")
    """
    if torch.cuda.is_available():
        # 如果有 NVIDIA GPU，优先使用 CUDA
        return "cuda"
    elif torch.backends.mps.is_available():
        # 如果是 Apple Silicon Mac，使用 MPS 加速
        return "mps"
    # 否则使用 CPU
    return "cpu"


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    执行一个训练轮次（epoch）

    Args:
        model: 要训练的模型
        dataloader: 训练数据加载器
        optimizer: 优化器（如 AdamW）
        criterion: 损失函数（如交叉熵）
        device: 计算设备

    Returns:
        tuple: (平均损失, 准确率)
    """
    model.train()  # 设置模型为训练模式（启用 Dropout 等）
    total_loss = 0  # 累计损失
    correct = 0  # 正确预测数
    total = 0  # 总样本数

    # 使用 tqdm 显示训练进度
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # 将数据移动到指定设备
        input_ids = batch["input_ids"].to(device)  # 输入的 token ID
        labels = batch["label"].to(device)  # 标签（0: 负面, 1: 正面）

        # 前向传播
        optimizer.zero_grad()  # 清除之前的梯度
        logits = model(input_ids)  # 模型预测，输出 logits
        loss = criterion(logits, labels)  # 计算损失

        # 反向传播
        loss.backward()  # 计算梯度
        # 梯度裁剪，防止梯度爆炸，最大梯度范数为 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # 更新模型参数

        # 统计指标
        total_loss += loss.item()  # 累加损失
        preds = logits.argmax(dim=-1)  # 取概率最大的类别作为预测
        correct += (preds == labels).sum().item()  # 统计正确预测数
        total += labels.size(0)  # 累加样本数

        # 实时更新进度条显示的损失和准确率
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

    # 返回平均损失和整体准确率
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """
    在验证/测试集上评估模型性能

    Args:
        model: 要评估的模型
        dataloader: 验证/测试数据加载器
        criterion: 损失函数
        device: 计算设备

    Returns:
        tuple: (平均损失, 准确率)
    """
    model.eval()  # 设置模型为评估模式（禁用 Dropout 等）
    total_loss = 0
    correct = 0
    total = 0

    # 评估时不需要计算梯度，节省内存和计算
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 将数据移动到指定设备
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            # 前向传播获取预测
            logits = model(input_ids)
            loss = criterion(logits, labels)

            # 统计指标
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)  # 取概率最大的类别
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    """
    主函数：完整的训练流程
    """
    # 1. 选择计算设备
    device = get_device()
    print(f"使用设备: {device}")

    # 2. 加载数据
    print("\n加载数据...")
    # load_data() 返回训练集、测试集和分词器
    train_dataset, test_dataset, tokenizer = load_data()
    # 创建数据加载器，用于批量加载数据
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset)

    # 3. 创建模型
    print("\n创建模型...")
    model = SentimentLLM(
        vocab_size=len(tokenizer.word2idx),  # 词表大小
        d_model=config.D_MODEL,  # 模型维度（嵌入维度）
        n_heads=config.N_HEADS,  # 多头注意力的头数
        n_layers=config.N_LAYERS,  # Transformer 层数
        d_ff=config.D_FF,  # 前馈网络隐藏层维度
        max_len=config.MAX_LENGTH,  # 最大序列长度
        num_classes=config.NUM_CLASSES,  # 分类数（2：正面/负面）
        dropout=config.DROPOUT  # Dropout 比例
    ).to(device)  # 将模型移动到指定设备

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 4. 配置优化器和损失函数
    # AdamW: 带权重衰减的 Adam 优化器，适合 Transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    # 交叉熵损失函数，适用于多分类任务
    criterion = nn.CrossEntropyLoss()

    # 5. 训练循环
    print("\n开始训练...")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")

        # 训练一个 epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        # 在测试集上评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 打印本轮结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 6. 保存训练好的模型
    # 只保存模型参数（state_dict），而不是整个模型对象
    torch.save(model.state_dict(), f"models/sentiment_model_{config.DATA_SIZE_NAME}.pt")
    print(f"\n模型已保存到 models/sentiment_model_{config.DATA_SIZE_NAME}.pt")


# 程序入口点
if __name__ == "__main__":
    main()
