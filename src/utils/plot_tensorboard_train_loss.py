#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "train_loss"  # 固定标量名

def find_event_files(root: Path) -> List[Path]:
    files = []
    for pat in ("**/events.out.tfevents.*", "**/events.tfevents.*"):
        files.extend([Path(p) for p in glob.glob(str(root / pat), recursive=True)])
    return sorted(set(files))

def run_name_firstdir(root: Path, event_path: Path) -> str:
    """把 root 下面的第一层目录名作为 run 名（如 default/no-position-embedding/...）。"""
    rel = event_path.relative_to(root)
    parts = list(rel.parts)
    return parts[0] if parts else event_path.parent.name

def read_scalar(event_file: Path, tag: str) -> Tuple[List[int], List[float]]:
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 10**9})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps = [ev.step for ev in events]
    vals = [float(ev.value) for ev in events]
    return steps, vals

def main():
    parser = argparse.ArgumentParser("Plot train_loss from multiple TensorBoard runs (single PDF).")
    parser.add_argument("--root", default="logs/train/translate",
                        help="包含事件文件的根目录（递归扫描）")
    parser.add_argument("--out", default="train_loss_compare.pdf",
                        help="输出 PDF 文件路径")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    files = find_event_files(root)

    # 收集数据：run -> (step, loss)
    collected: Dict[str, List[Tuple[int, float]]] = {}
    for f in files:
        steps, vals = read_scalar(f, TAG)
        if not steps:
            continue
        run = run_name_firstdir(root, f)
        collected.setdefault(run, []).extend(zip(steps, vals))

    # 转成 DataFrame，并按 step 排序/去重
    run_dfs: Dict[str, pd.DataFrame] = {}
    for run, pairs in collected.items():
        if not pairs:
            continue
        df = pd.DataFrame(pairs, columns=["step", "loss"])
        df = df.drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)
        run_dfs[run] = df

    if not run_dfs:
        print(f"[!] 在 {root} 未找到 tag='{TAG}' 的标量数据。")
        return

    # 画单张图（不指定颜色，使用 matplotlib 默认）
    plt.figure(figsize=(9, 5), dpi=160)
    for run, df in sorted(run_dfs.items()):
        if df.empty:
            continue
        plt.plot(df["step"], df["loss"], label=run, linewidth=1.8)

    plt.title(f"Training Loss")
    plt.xlabel("Global Step")
    plt.ylabel("Training Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)  # 只保存 PDF
    print(f"[+] 已保存：{out_path}")

if __name__ == "__main__":
    main()
