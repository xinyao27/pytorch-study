#!/usr/bin/env bun
import { spawn } from "child_process";
import * as p from "@clack/prompts";
import pc from "picocolors";

async function runInference(text) {
  return new Promise((resolve, reject) => {
    const py = spawn("uv", ["run", "python", "-c", `
import sys
sys.path.insert(0, '.')
from inference import load_model, predict
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, tokenizer = load_model()
model = model.to(device)

text = """${text.replace(/"/g, '\\"')}"""
result = predict(text, model, tokenizer, device)
print(f"{result['sentiment']}|{result['confidence']:.4f}|{result['scores']['消极']:.4f}|{result['scores']['积极']:.4f}")
`], { cwd: import.meta.dirname });

    let output = "";
    let error = "";

    py.stdout.on("data", (data) => {
      output += data.toString();
    });

    py.stderr.on("data", (data) => {
      error += data.toString();
    });

    py.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(error || "Inference failed"));
        return;
      }
      const [sentiment, confidence, negScore, posScore] = output.trim().split("|");
      resolve({
        sentiment,
        confidence: parseFloat(confidence),
        scores: {
          negative: parseFloat(negScore),
          positive: parseFloat(posScore),
        },
      });
    });
  });
}

function formatResult(result) {
  const isPositive = result.sentiment === "积极";
  const icon = isPositive ? pc.green("●") : pc.red("●");
  const sentiment = isPositive ? pc.green(result.sentiment) : pc.red(result.sentiment);
  const confidence = pc.cyan(`${(result.confidence * 100).toFixed(1)}%`);

  const barWidth = 20;
  const posWidth = Math.round(result.scores.positive * barWidth);
  const negWidth = barWidth - posWidth;
  const bar = pc.red("█".repeat(negWidth)) + pc.green("█".repeat(posWidth));

  return `
${icon} ${pc.bold("情感")}: ${sentiment}  ${pc.dim("置信度")}: ${confidence}

${pc.dim("消极")} ${bar} ${pc.dim("积极")}
${pc.dim(`     ${(result.scores.negative * 100).toFixed(1)}%`.padEnd(barWidth + 2))}${pc.dim(`${(result.scores.positive * 100).toFixed(1)}%`)}
`;
}

async function main() {
  console.clear();

  p.intro(pc.bgCyan(pc.black(" 情感分析 CLI ")));

  const s = p.spinner();
  s.start("加载模型...");

  // 预热模型
  try {
    await runInference("测试");
    s.stop("模型加载完成");
  } catch (e) {
    s.stop("模型加载失败");
    p.log.error(e.message);
    p.note("请确保已运行:\nuv run python dataset.py\nuv run python train.py", "提示");
    p.outro(pc.red("退出"));
    process.exit(1);
  }

  while (true) {
    const text = await p.text({
      message: "输入要分析的文本",
      placeholder: "今天心情很好...",
      validate: (value) => {
        if (!value.trim()) return "请输入文本";
      },
    });

    if (p.isCancel(text)) {
      p.outro(pc.dim("再见!"));
      break;
    }

    const s = p.spinner();
    s.start("分析中...");

    try {
      const result = await runInference(text);
      s.stop("分析完成");
      console.log(formatResult(result));
    } catch (e) {
      s.stop("分析失败");
      p.log.error(e.message);
    }
  }
}

main().catch(console.error);
