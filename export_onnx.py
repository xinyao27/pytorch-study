"""导出 ONNX 模型用于 Netron 可视化"""

import torch
from model import SentimentLLM
from tokenizer import SimpleTokenizer
import config


def export_onnx(model_path=None, output_path=None):
    if model_path is None:
        model_path = f"models/sentiment_model_{config.DATA_SIZE_NAME}.pt"
    if output_path is None:
        output_path = f"models/sentiment_model_{config.DATA_SIZE_NAME}.onnx"

    # 加载分词器获取词表大小
    tokenizer = SimpleTokenizer(vocab_size=config.VOCAB_SIZE)
    tokenizer.load(f"data/vocab_{config.DATA_SIZE_NAME}.json")

    # 创建模型
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

    # 创建示例输入
    dummy_input = torch.randint(0, len(tokenizer.word2idx), (1, config.MAX_LENGTH))

    # 导出 ONNX (使用旧版导出方式)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=14,
        dynamo=False  # 使用旧版导出
    )

    print(f"ONNX 模型已导出: {output_path}")
    print(f"使用 Netron 打开: npx netron {output_path}")


if __name__ == "__main__":
    export_onnx()
