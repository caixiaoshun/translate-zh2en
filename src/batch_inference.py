# gen_csv.py
import argparse
import json
import os
from typing import List

import pandas as pd
import torch
from torch import nn
from omegaconf import OmegaConf
import hydra
from transformers import AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/translation2019zh_valid.json", help="validation jsonl 路径")
    p.add_argument("--csv_out", required=True, help="输出 CSV 路径")
    p.add_argument("--config", default="configs/model/translate.yaml")
    p.add_argument("--weight_path", default="logs/train/translate/default/2025-10-19_09-59-13/checkpoints/last.ckpt")
    p.add_argument("--tokenizer_path", default="checkpoints/mt5-small")
    p.add_argument("--max_src_len", type=int, default=128)
    p.add_argument("--max_tgt_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def build_module(config_path: str, weight_path: str) -> nn.Module:
    cfg = OmegaConf.load(config_path)["net"]
    model: nn.Module = hydra.utils.instantiate(cfg)
    state = torch.load(weight_path, map_location="cpu", weights_only=False)["state_dict"]
    new_state = {}
    for k, v in state.items():
    
        nk = k[len("net._orig_mod."):] if k.startswith("net._orig_mod.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)
    
    model.eval().to(DEVICE)
    return model

def build_tokenizer(tokenizer_path: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    return tok


def build_encoder_mask(src_ids: torch.Tensor, pad_id: int) -> torch.Tensor:

    device = src_ids.device
    neg_inf = -1e8
    visible = (src_ids != pad_id)[:, None, None, :]
    return torch.where(visible, torch.tensor(0.0, device=device), torch.tensor(neg_inf, device=device))

def build_decoder_mask(tgt_in_ids: torch.Tensor, pad_id: int) -> torch.Tensor:

    device = tgt_in_ids.device
    B, T = tgt_in_ids.shape
    neg_inf = -1e8

    causal = torch.zeros((T, T), device=device)
    causal = causal.masked_fill(torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1), neg_inf)
    causal = causal[None, None, :, :]  # [1,1,T,T]

    key_visible = (tgt_in_ids != pad_id)[:, None, None, :]   # [B,1,1,T]
    key_mask = torch.where(key_visible, torch.tensor(0.0, device=device), torch.tensor(neg_inf, device=device))

    query_visible = (tgt_in_ids != pad_id)[:, None, :, None] # [B,1,T,1]
    query_mask = torch.where(query_visible, torch.tensor(0.0, device=device), torch.tensor(neg_inf, device=device))

    return causal + key_mask + query_mask


@torch.no_grad()
def greedy_decode(model, src_ids: torch.Tensor, pad_id: int, eos_id: int, max_len: int) -> torch.Tensor:
    # src_ids: [B,S]
    enc_mask = build_encoder_mask(src_ids, pad_id=pad_id)
    src_hidden = model.embedding(src_ids)
    for layer in model.encoder:
        src_hidden = layer(src_hidden, enc_mask)
    memory = src_hidden

    B = src_ids.size(0)
    out = torch.full((B, 1), pad_id, dtype=torch.long, device=src_ids.device)
    for _ in range(max_len - 1):
        dec_mask = build_decoder_mask(out, pad_id=pad_id)
        tgt_hidden = model.embedding(out)
        for layer in model.decoder:
            tgt_hidden = layer(tgt_hidden, memory, enc_mask, dec_mask)
        logits = model.head(tgt_hidden)
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, next_token], dim=1)
        if (next_token == eos_id).all():
            break
    return out  # [B,T]

# ----------------------------
# 3) 编码 & 批量推理
# ----------------------------
def encode_batch(tokenizer, texts: List[str], max_src_len: int) -> torch.Tensor:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    seqs = []
    for t in texts:
        ids = tokenizer.encode(str(t), add_special_tokens=False)
        ids = ids[: max_src_len - 1] + [eos_id]
        seqs.append(torch.tensor(ids, dtype=torch.long))
    max_len = max(x.size(0) for x in seqs)
    batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(seqs):
        batch[i, : x.size(0)] = x
    return batch.to(DEVICE)

def decode_batch(tokenizer, out_ids: torch.Tensor) -> List[str]:
    eos_id = tokenizer.eos_token_id
    texts = []
    for seq in out_ids:  # [T]
        toks = seq[1:].tolist()
        if eos_id in toks:
            toks = toks[: toks.index(eos_id) + 1]
        text = tokenizer.decode(toks, skip_special_tokens=True)
        texts.append(text)
    return texts

def translate_corpus(model, tokenizer, src_texts: List[str], max_src_len: int, max_tgt_len: int, batch_size: int) -> List[str]:
    preds = []
    N = len(src_texts)
    for i in tqdm(range(0, N, batch_size), total=N // batch_size):
        batch_texts = src_texts[i : i + batch_size]
        src_ids = encode_batch(tokenizer, batch_texts, max_src_len)    # [B,S]
        out_ids = greedy_decode(model, src_ids, tokenizer.pad_token_id, tokenizer.eos_token_id, max_tgt_len)  # [B,T]
        batch_preds = decode_batch(tokenizer, out_ids)
        preds.extend(batch_preds)

    return preds

# ----------------------------
# 4) I/O
# ----------------------------
def read_jsonl(path: str):
    srcs, tgts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = obj['chinese']
            tgt = obj['english']
            srcs.append(str(src))
            tgts.append(str(tgt))
    return srcs, tgts

def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    print("Loading model/tokenizer...")
    model = build_module(args.config, args.weight_path)
    tokenizer = build_tokenizer(args.tokenizer_path)

    print("Reading jsonl...")
    src_texts, gts = read_jsonl(args.jsonl)
    assert len(src_texts) == len(gts) and len(gts) > 0, "数据为空或长度不一致"

    print(f"Translating {len(src_texts)} samples ...")
    preds = translate_corpus(
        model, tokenizer,
        src_texts, args.max_src_len, args.max_tgt_len,
        args.batch_size
    )

    df = pd.DataFrame({"pred": preds, "gt": gts})
    df.to_csv(args.csv_out, index=False, encoding="utf-8")
    print("Saved to:", args.csv_out)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
