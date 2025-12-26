#!/usr/bin/env bun
import { spawn } from "child_process";

async function runInference(text) {
  return new Promise((resolve, reject) => {
    const py = spawn("uv", ["run", "python", "-c", `
import sys
sys.path.insert(0, '.')
try:
    from inference import load_model, predict
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer = load_model()
    model = model.to(device)

    text = """${text.replace(/"/g, '\\"').replace(/`/g, '\\`')}"""
    result = predict(text, model, tokenizer, device)
    print(f"{result['sentiment']}|{result['confidence']:.4f}|{result['scores']['消极']:.4f}|{result['scores']['积极']:.4f}")
except Exception as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)
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
      // 取最后一行有效输出（忽略警告等）
      const lines = output.trim().split("\n");
      const lastLine = lines[lines.length - 1];
      const parts = lastLine.split("|");
      if (parts.length !== 4) {
        reject(new Error("解析失败: " + output));
        return;
      }
      const [sentiment, confidence, negScore, posScore] = parts;
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

const html = `<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>情感分析</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .container {
      width: 100%;
      max-width: 500px;
      padding: 2rem;
    }
    h1 {
      text-align: center;
      margin-bottom: 2rem;
      font-size: 1.5rem;
      color: #22d3ee;
    }
    textarea {
      width: 100%;
      height: 120px;
      padding: 1rem;
      border: 1px solid #334155;
      border-radius: 8px;
      background: #1e293b;
      color: #e2e8f0;
      font-size: 1rem;
      resize: none;
      outline: none;
    }
    textarea:focus { border-color: #22d3ee; }
    button {
      width: 100%;
      padding: 0.875rem;
      margin-top: 1rem;
      border: none;
      border-radius: 8px;
      background: #22d3ee;
      color: #0f172a;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: #06b6d4; }
    button:disabled { background: #475569; cursor: not-allowed; }
    .result {
      margin-top: 1.5rem;
      padding: 1.5rem;
      border-radius: 8px;
      background: #1e293b;
      display: none;
    }
    .result.show { display: block; }
    .sentiment {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 1.25rem;
      font-weight: 600;
    }
    .sentiment .dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
    }
    .positive .dot { background: #22c55e; }
    .negative .dot { background: #ef4444; }
    .uncertain .dot { background: #f59e0b; }
    .positive .label { color: #22c55e; }
    .negative .label { color: #ef4444; }
    .uncertain .label { color: #f59e0b; }
    .confidence {
      margin-top: 0.5rem;
      color: #94a3b8;
      font-size: 0.875rem;
    }
    .bar-container {
      margin-top: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.75rem;
      color: #64748b;
    }
    .bar {
      flex: 1;
      height: 8px;
      border-radius: 4px;
      background: #334155;
      overflow: hidden;
      display: flex;
    }
    .bar-neg { background: #ef4444; }
    .bar-pos { background: #22c55e; }
  </style>
</head>
<body>
  <div class="container">
    <h1>情感分析</h1>
    <textarea id="input" placeholder="输入要分析的文本..."></textarea>
    <button id="submit">分析</button>
    <div class="result" id="result">
      <div class="sentiment" id="sentiment">
        <span class="dot"></span>
        <span class="label"></span>
      </div>
      <div class="confidence" id="confidence"></div>
      <div class="bar-container">
        <span>消极</span>
        <div class="bar">
          <div class="bar-neg" id="barNeg"></div>
          <div class="bar-pos" id="barPos"></div>
        </div>
        <span>积极</span>
      </div>
    </div>
  </div>
  <script>
    const input = document.getElementById('input');
    const btn = document.getElementById('submit');
    const result = document.getElementById('result');
    const sentiment = document.getElementById('sentiment');
    const confidence = document.getElementById('confidence');
    const barNeg = document.getElementById('barNeg');
    const barPos = document.getElementById('barPos');

    btn.onclick = async () => {
      const text = input.value.trim();
      if (!text) return;

      btn.disabled = true;
      btn.textContent = '分析中...';

      try {
        const res = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        const data = await res.json();

        if (data.error) {
          throw new Error(data.error);
        }
        if (!data.scores) {
          throw new Error('响应格式错误: ' + JSON.stringify(data));
        }

        const isPositive = data.sentiment === '积极';
        const isUncertain = data.sentiment === '不确定';
        sentiment.className = 'sentiment ' + (isUncertain ? 'uncertain' : isPositive ? 'positive' : 'negative');
        sentiment.querySelector('.label').textContent = data.sentiment;
        confidence.textContent = '置信度: ' + (data.confidence * 100).toFixed(1) + '%';
        barNeg.style.width = (data.scores.negative * 100) + '%';
        barPos.style.width = (data.scores.positive * 100) + '%';
        result.classList.add('show');
      } catch (e) {
        alert('分析失败: ' + e.message);
      }

      btn.disabled = false;
      btn.textContent = '分析';
    };

    input.onkeydown = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        btn.click();
      }
    };
  </script>
</body>
</html>`;

const server = Bun.serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/" && req.method === "GET") {
      return new Response(html, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }

    if (url.pathname === "/analyze" && req.method === "POST") {
      try {
        const { text } = await req.json();
        if (!text?.trim()) {
          return Response.json({ error: "请输入文本" }, { status: 400 });
        }
        const result = await runInference(text);
        return Response.json(result);
      } catch (e) {
        return Response.json({ error: e.message }, { status: 500 });
      }
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`Server running at http://localhost:${server.port}`);
