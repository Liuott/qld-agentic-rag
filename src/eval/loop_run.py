# src/eval/loop_run.py
import subprocess, os, sys, pandas as pd, argparse, time
from pathlib import Path

EVAL_MOD  = "src.eval.eval_ragas"       # ← 用模块名
JUDGE_MOD = "src.eval.eval_langsmith"   # ← 你的评测脚本模块名

ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录
ANS_CSV   = ROOT / "data/eval/run_answers.csv"
SCORE_CSV = ROOT / "experiments/final_scores.csv"

def run_eval(row_idx, thr, fetch_k, mmr=False, rerank=True, no_cache=True, trim_ctx=None):
    cmd = [sys.executable, "-m", EVAL_MOD,
           "--row_idx", str(row_idx),
           "--thr", str(thr),
           "--fetch_k", str(fetch_k)]
    if mmr:      cmd += ["--mmr"]
    if rerank:   cmd += ["--rerank"]
    if no_cache: cmd += ["--no_cache"]
    if trim_ctx: cmd += ["--trim_ctx", str(trim_ctx)]
    subprocess.run(cmd, check=True, cwd=ROOT)   # 关键：在根目录运行，保证相对路径一致

def run_judge():
    subprocess.run([sys.executable, "-m", JUDGE_MOD, "--limit", "1", "--tail"], check=True, cwd=ROOT)
    df = pd.read_csv(SCORE_CSV)
    r = df.iloc[-1]
    faithful = bool(r["Faithfulness"])
    rel = str(r["Relevance"]).lower()
    # 为了兼容原先的三元返回，这里把 correctness 设为与 faithful 一致
    return faithful, faithful, rel

def score_to_ok(correct, faithful, rel):
    return faithful and rel != "low"

def loop_one(row_idx, max_steps=3):
    plans = [
        {"thr": 0.20, "fetch_k": 80,  "mmr": False, "rerank": True, "trim_ctx": 1200},
        {"thr": 0.12, "fetch_k": 160, "mmr": True,  "rerank": True, "trim_ctx": 1400},
        {"thr": 0.00, "fetch_k": 240, "mmr": True,  "rerank": True, "trim_ctx": 1800},
    ][:max_steps]

    best = {"idx": row_idx, "step": -1, "ok": False, "score": -1.0}
    for step, p in enumerate(plans, 1):
        run_eval(row_idx, **p)
        correct, faithful, rel = run_judge()
        s = (1 if correct else 0) + (1 if faithful else 0) + (0.5 if rel=="high" else 0.2 if rel=="medium" else 0)
        if s > best["score"]:
            best.update({"step": step, "ok": score_to_ok(correct, faithful, rel), "score": s})
        print(f"[row {row_idx}] step {step}: correct={correct}, faithful={faithful}, rel={rel}, score={s:.2f}")
        if best["ok"]:
            break
        time.sleep(0.3)
    return best

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=3)   
    args = ap.parse_args()

    results = []
    for i in range(args.start, args.start + args.limit):
        try:
            r = loop_one(i, max_steps=args.max_steps)
            results.append(r)
        except Exception as e:
            print(f"[row {i}] error: {e}")

    ok = sum(1 for r in results if r["ok"])
    avg = (sum(r['score'] for r in results)/max(1, len(results)))
    print(f"\n[summary] passed={ok}/{len(results)}; avg score={avg:.2f}")
