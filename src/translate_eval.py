# eval_mt.py
import argparse
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF, TER

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="输入CSV路径，包含 pred 和 gt 两列")
    return ap.parse_args()

def main():
    args = parse_args()

    df = pd.read_csv(args.csv)

    sys = df["pred"].astype(str).fillna("").tolist()
    ref = df["gt"].astype(str).fillna("").tolist()
    refs = [ref]

    # 指标
    bleu = BLEU(tokenize="13a", lowercase=True, effective_order=True)
    chrf = CHRF(beta=2, char_order=6, word_order=0)
    ter = TER()

    bleu_score = bleu.corpus_score(sys, refs)
    chrf_score = chrf.corpus_score(sys, refs)
    ter_score = ter.corpus_score(sys, refs)

    print("====== MT Evaluation (sacreBLEU) ======")
    print(f"Samples: {len(sys)}")
    print(f"BLEU: {bleu_score.score:.2f}")
    print(f"chrF: {chrf_score.score:.2f}")
    print(f"TER:  {ter_score.score:.2f}")
    print("BLEU signature:", bleu.get_signature())
    
if __name__ == "__main__":
    main()
