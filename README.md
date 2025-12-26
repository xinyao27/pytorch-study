# Sentiment LLM

基于 PyTorch 的情感分类 Transformer 模型，判断一句话是积极还是消极。

## 项目结构

```
├── config.py        # 配置参数
├── tokenizer.py     # 简易词级分词器
├── dataset.py       # 数据集准备和加载
├── model.py         # Transformer 模型架构
├── train.py         # 训练脚本
├── inference.py     # 推理脚本
└── export_onnx.py   # ONNX 模型导出
```

## 模型架构

```
输入文本 → Tokenizer → Embedding → PositionalEncoding
                                        ↓
                              4 × TransformerBlock
                              (MultiheadAttention + FFN)
                                        ↓
                              CLS Pooling → Classifier → 积极/消极
```

- 参数量: ~4.7M
- 词表大小: 10,000
- 隐藏维度: 256
- 注意力头数: 4
- Transformer 层数: 4

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 准备数据

下载 IMDB 数据集并构建词表：

```bash
uv run python dataset.py
```

输出文件：
- `data.pt` - 处理后的数据
- `vocab.json` - 词表

### 3. 训练模型

```bash
uv run python train.py
```

输出文件：
- `sentiment_model.pt` - 训练好的模型

### 4. 测试推理

批量测试：
```bash
uv run python inference.py
```

交互式测试：
```bash
uv run python inference.py -i
```

然后输入英文句子进行情感分析：
```
> I really enjoyed this movie
预测: 积极 ✓ (97.9%)

> This is terrible
预测: 消极 ✗ (85.3%)

> q
Bye!
```

### 5. 导出 ONNX（可选）

导出模型用于 Netron 可视化：

```bash
uv run python export_onnx.py
```

查看模型结构：
```bash
uv run netron sentiment_model.onnx
```

浏览器打开 http://localhost:8080

## 配置参数

在 `config.py` 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| TRAIN_SIZE | 20000 | 训练数据量 |
| TEST_SIZE | 2000 | 测试数据量 |
| EPOCHS | 5 | 训练轮数 |
| LR | 3e-4 | 学习率 |
| D_MODEL | 256 | 隐藏维度 |
| N_LAYERS | 4 | Transformer 层数 |

## 训练效果

使用 20,000 条 IMDB 电影评论训练 5 个 epoch：

| Epoch | Train Acc | Test Acc |
|-------|-----------|----------|
| 1 | 67.5% | 74.6% |
| 3 | 77.5% | 77.8% |
| 5 | 80.0% | 77.5% |
