"""模型 Benchmark 评估系统"""

import argparse
import base64
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

import config
from model import SentimentLLM
from tokenizer import SimpleTokenizer


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path, vocab_path=None):
    """加载模型和分词器"""
    if vocab_path is None:
        vocab_path = f"data/vocab_{config.DATA_SIZE_NAME}.json"

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
        dropout=config.DROPOUT,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model, tokenizer


def load_test_data(data_path=None):
    """加载测试数据"""
    if data_path is None:
        data_path = f"data/test_data_{config.DATA_SIZE_NAME}.jsonl"

    test_data = []
    with open(data_path, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data


def predict_batch(texts, model, tokenizer, device):
    """批量预测，返回概率和预测结果"""
    model.eval()
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer.encode(text, config.MAX_LENGTH)
            tokens = torch.tensor([encoded]).to(device)
            logits = model(tokens)
            probs = torch.softmax(logits, dim=-1)[0]
            pred = logits.argmax(dim=-1).item()

            all_probs.append(probs.cpu().numpy())
            all_preds.append(pred)

    return np.array(all_probs), np.array(all_preds)


def compute_metrics(y_true, y_pred, y_probs):
    """计算所有分类指标"""
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC-ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    auc_roc = auc(fpr, tpr)

    # AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs[:, 1])
    auc_pr = auc(recall_curve, precision_curve)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "roc_curve": (fpr, tpr),
        "pr_curve": (recall_curve, precision_curve),
    }


def measure_inference_speed(model, tokenizer, device, num_samples=100):
    """测量推理速度"""
    # 生成测试文本
    test_text = "这是一个用于测试推理速度的示例文本，包含一些中文字符。"

    model.eval()
    encoded = tokenizer.encode(test_text, config.MAX_LENGTH)
    tokens = torch.tensor([encoded]).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(tokens)

    # 计时
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_samples):
            _ = model(tokens)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    return num_samples / elapsed


def analyze_confidence(y_true, y_pred, y_probs):
    """置信度分析"""
    confidences = np.max(y_probs, axis=1)
    correct = y_true == y_pred

    # 按置信度区间统计
    bins = [
        ("高 (>0.9)", confidences > 0.9),
        ("中 (0.7-0.9)", (confidences >= 0.7) & (confidences <= 0.9)),
        ("低 (<0.7)", confidences < 0.7),
    ]

    analysis = []
    for name, mask in bins:
        if mask.sum() > 0:
            acc = correct[mask].mean()
            count = mask.sum()
            analysis.append({"range": name, "count": int(count), "accuracy": float(acc)})
        else:
            analysis.append({"range": name, "count": 0, "accuracy": 0.0})

    return analysis, confidences


def get_model_info(model, model_path):
    """获取模型信息"""
    num_params = sum(p.numel() for p in model.parameters())
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

    return {"num_params": num_params, "file_size_mb": file_size}


# ==================== 可视化函数 ====================


def fig_to_base64(fig):
    """将 matplotlib 图转换为 base64 字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def plot_confusion_matrix(y_true, y_pred, model_name="model"):
    """绘制混淆矩阵，返回 base64"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["消极", "积极"],
        yticklabels=["消极", "积极"],
        ax=ax,
    )
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(f"混淆矩阵 - {model_name}")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_roc_curve(metrics, model_name="model"):
    """绘制 ROC 曲线，返回 base64"""
    fpr, tpr = metrics["roc_curve"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f'AUC = {metrics["auc_roc"]:.4f}')
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机猜测")
    ax.set_xlabel("假正率 (FPR)")
    ax.set_ylabel("真正率 (TPR)")
    ax.set_title(f"ROC 曲线 - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_pr_curve(metrics, model_name="model"):
    """绘制 PR 曲线，返回 base64"""
    recall, precision = metrics["pr_curve"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, "b-", linewidth=2, label=f'AUC = {metrics["auc_pr"]:.4f}')
    ax.set_xlabel("召回率 (Recall)")
    ax.set_ylabel("精确率 (Precision)")
    ax.set_title(f"PR 曲线 - {model_name}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_confidence_distribution(confidences, y_true, y_pred, model_name="model"):
    """绘制置信度分布，返回 base64"""
    correct = y_true == y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidences[correct], bins=20, alpha=0.7, label="正确预测", color="green")
    ax.hist(confidences[~correct], bins=20, alpha=0.7, label="错误预测", color="red")
    ax.set_xlabel("置信度")
    ax.set_ylabel("样本数")
    ax.set_title(f"置信度分布 - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_model_comparison(results):
    """绘制多模型对比图，返回 base64"""
    if len(results) < 2:
        return None

    model_names = list(results.keys())
    metrics_to_compare = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    x = np.arange(len(metrics_to_compare))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        values = [results[name]["metrics"][m] for m in metrics_to_compare]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_xlabel("指标")
    ax.set_ylabel("得分")
    ax.set_title("模型对比")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


# ==================== 报告输出 ====================


def print_report(model_name, metrics, model_info, speed, confidence_analysis, data_info):
    """打印评估报告"""
    print("\n" + "═" * 55)
    print("              Benchmark Report")
    print("═" * 55)
    print(f"Model: {model_name}")
    print(f"Data:  {data_info['path']} ({data_info['count']} samples)")
    print("─" * 55)
    print("Classification Metrics:")
    print(f"  Accuracy:   {metrics['accuracy']:.2%}")
    print(f"  Precision:  {metrics['precision']:.2%}")
    print(f"  Recall:     {metrics['recall']:.2%}")
    print(f"  F1 Score:   {metrics['f1']:.2%}")
    print(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:     {metrics['auc_pr']:.4f}")
    print("─" * 55)
    print("Performance:")
    print(f"  Inference Speed:  {speed:.1f} samples/sec")
    print(f"  Model Parameters: {model_info['num_params'] / 1e6:.2f}M")
    print(f"  Model Size:       {model_info['file_size_mb']:.2f} MB")
    print("─" * 55)
    print("Confidence Analysis:")
    for item in confidence_analysis:
        print(f"  {item['range']:15} {item['count']:5} samples, {item['accuracy']:.1%} accuracy")
    print("═" * 55)


def generate_html_report(results, data_info, output_path):
    """生成 HTML 报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构建模型卡片 HTML
    model_cards = []
    for name, data in results.items():
        m = data["metrics"]
        info = data["model_info"]
        conf = data["confidence_analysis"]
        plots = data["plots"]

        card = f"""
        <div class="model-card">
            <h2>{name}</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{m['accuracy']:.1%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{m['precision']:.1%}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{m['recall']:.1%}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{m['f1']:.1%}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{m['auc_roc']:.4f}</div>
                    <div class="metric-label">AUC-ROC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{m['auc_pr']:.4f}</div>
                    <div class="metric-label">AUC-PR</div>
                </div>
            </div>

            <h3>性能指标</h3>
            <table class="info-table">
                <tr><td>推理速度</td><td>{data['speed']:.1f} samples/sec</td></tr>
                <tr><td>模型参数量</td><td>{info['num_params'] / 1e6:.2f}M</td></tr>
                <tr><td>模型大小</td><td>{info['file_size_mb']:.2f} MB</td></tr>
            </table>

            <h3>置信度分析</h3>
            <table class="confidence-table">
                <tr><th>置信度区间</th><th>样本数</th><th>准确率</th></tr>
                {"".join(f"<tr><td>{c['range']}</td><td>{c['count']}</td><td>{c['accuracy']:.1%}</td></tr>" for c in conf)}
            </table>

            <h3>可视化</h3>
            <div class="plots-grid">
                <div class="plot">
                    <img src="data:image/png;base64,{plots['confusion_matrix']}" alt="混淆矩阵">
                </div>
                <div class="plot">
                    <img src="data:image/png;base64,{plots['roc_curve']}" alt="ROC曲线">
                </div>
                <div class="plot">
                    <img src="data:image/png;base64,{plots['pr_curve']}" alt="PR曲线">
                </div>
                <div class="plot">
                    <img src="data:image/png;base64,{plots['confidence_dist']}" alt="置信度分布">
                </div>
            </div>
        </div>
        """
        model_cards.append(card)

    # 多模型对比图
    comparison_html = ""
    if len(results) > 1:
        comparison_b64 = plot_model_comparison(results)
        if comparison_b64:
            comparison_html = f"""
            <div class="comparison-section">
                <h2>模型对比</h2>
                <img src="data:image/png;base64,{comparison_b64}" alt="模型对比">
            </div>
            """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }}
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        header p {{
            opacity: 0.9;
            font-size: 1.1rem;
        }}
        .model-card {{
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .model-card h2 {{
            color: #333;
            font-size: 1.8rem;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}
        .model-card h3 {{
            color: #555;
            font-size: 1.2rem;
            margin: 25px 0 15px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        .info-table, .confidence-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        .info-table td, .confidence-table td, .confidence-table th {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        .confidence-table th {{
            background: #f8f9fa;
            text-align: left;
            font-weight: 600;
        }}
        .info-table td:first-child {{
            color: #666;
            width: 40%;
        }}
        .info-table td:last-child {{
            font-weight: 600;
            color: #333;
        }}
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .plot {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        .comparison-section {{
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .comparison-section h2 {{
            color: #333;
            margin-bottom: 20px;
        }}
        .comparison-section img {{
            max-width: 100%;
            border-radius: 8px;
        }}
        footer {{
            text-align: center;
            color: white;
            opacity: 0.8;
            margin-top: 20px;
            font-size: 0.9rem;
        }}
        @media (max-width: 768px) {{
            .plots-grid {{
                grid-template-columns: 1fr;
            }}
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Benchmark Report</h1>
            <p>测试数据: {data_info['path']} ({data_info['count']} samples) | 生成时间: {timestamp}</p>
        </header>

        {comparison_html}

        {"".join(model_cards)}

        <footer>
            <p>Generated by PyTorch Sentiment Model Benchmark System</p>
        </footer>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def save_report_json(results, output_dir):
    """保存 JSON 报告"""
    # 移除不可序列化的数据
    report = {}
    for name, data in results.items():
        report[name] = {
            "metrics": {k: v for k, v in data["metrics"].items() if k not in ["roc_curve", "pr_curve"]},
            "model_info": data["model_info"],
            "speed": data["speed"],
            "confidence_analysis": data["confidence_analysis"],
        }

    with open(output_dir / "benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ==================== 主函数 ====================


def evaluate_model(model_path, test_data, device):
    """评估单个模型"""
    model_name = Path(model_path).stem
    print(f"\n评估模型: {model_name}")
    print("加载模型...")

    model, tokenizer = load_model(model_path)
    model = model.to(device)

    # 提取数据
    texts = [item["text"] for item in test_data]
    y_true = np.array([item["label"] for item in test_data])

    # 预测
    print("运行预测...")
    y_probs, y_pred = predict_batch(texts, model, tokenizer, device)

    # 计算指标
    print("计算指标...")
    metrics = compute_metrics(y_true, y_pred, y_probs)

    # 性能测试
    print("测量推理速度...")
    speed = measure_inference_speed(model, tokenizer, device)

    # 置信度分析
    confidence_analysis, confidences = analyze_confidence(y_true, y_pred, y_probs)

    # 模型信息
    model_info = get_model_info(model, model_path)

    # 生成可视化（返回 base64）
    print("生成可视化...")
    plots = {
        "confusion_matrix": plot_confusion_matrix(y_true, y_pred, model_name),
        "roc_curve": plot_roc_curve(metrics, model_name),
        "pr_curve": plot_pr_curve(metrics, model_name),
        "confidence_dist": plot_confidence_distribution(confidences, y_true, y_pred, model_name),
    }

    return {
        "metrics": metrics,
        "model_info": model_info,
        "speed": speed,
        "confidence_analysis": confidence_analysis,
        "plots": plots,
    }


def main():
    parser = argparse.ArgumentParser(description="模型 Benchmark 评估系统")
    parser.add_argument("--model", type=str, help="模型文件路径")
    parser.add_argument("--compare", nargs="+", help="多模型对比，指定多个模型路径")
    parser.add_argument("--data", type=str, help="测试数据路径")
    parser.add_argument("--output", type=str, default="reports", help="输出目录")
    args = parser.parse_args()

    # 设备
    device = get_device()
    print(f"使用设备: {device}")

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # 加载测试数据
    test_data = load_test_data(args.data)
    data_info = {
        "path": args.data or f"data/test_data_{config.DATA_SIZE_NAME}.jsonl",
        "count": len(test_data),
    }

    # 确定要评估的模型
    if args.compare:
        model_paths = args.compare
    elif args.model:
        model_paths = [args.model]
    else:
        model_paths = [f"models/sentiment_model_{config.DATA_SIZE_NAME}.pt"]

    # 评估所有模型
    results = {}
    for model_path in model_paths:
        model_name = Path(model_path).stem
        results[model_name] = evaluate_model(model_path, test_data, device)
        print_report(
            model_name,
            results[model_name]["metrics"],
            results[model_name]["model_info"],
            results[model_name]["speed"],
            results[model_name]["confidence_analysis"],
            data_info,
        )

    # 生成 HTML 报告
    html_path = output_dir / "benchmark_report.html"
    generate_html_report(results, data_info, html_path)

    # 保存 JSON 报告
    save_report_json(results, output_dir)

    print(f"\n报告已保存到: {html_path}")


if __name__ == "__main__":
    main()
