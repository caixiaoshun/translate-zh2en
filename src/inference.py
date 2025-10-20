# infer.py
import argparse
import torch
from torch import nn
from omegaconf import OmegaConf
import hydra
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/model/translate.yaml")
    p.add_argument("--weight_path", default="logs/train/translate/default/2025-10-19_09-59-13/checkpoints/last.ckpt")
    p.add_argument("--tokenizer_path", default="checkpoints/mt5-small")
    p.add_argument("--text", default="今天天气不错，我们去公园散步吧。", help="要翻译的中文")
    p.add_argument("--max_src_len", type=int, default=128)
    p.add_argument("--max_tgt_len", type=int, default=128)
    return p.parse_args()


def build_module(config_path: str, weight_path: str) -> nn.Module:
    cfg = OmegaConf.load(config_path)["net"]
    model: nn.Module = hydra.utils.instantiate(cfg)

    state = torch.load(weight_path, map_location="cpu", weights_only=False)['state_dict']

    new_state = {}
    for k, v in state.items(): 
        nk = k[len("net._orig_mod."):]
        new_state[nk] = v

    model.load_state_dict(new_state, strict=True)

    model.eval().to(DEVICE)
    return model


# ----------------------------
# 3) 分词器
# ----------------------------
def build_tokenizer(tokenizer_path: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    return tok


# ----------------------------
# 4) 掩码工具（与 DataModule 保持一致）
# ----------------------------
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


# ----------------------------
# 5) 编码 + 贪婪解码
# ----------------------------
@torch.no_grad()
def greedy_decode(model, src_ids: torch.Tensor, pad_id: int, eos_id: int, max_len: int) -> torch.Tensor:
    model.eval()
    device = src_ids.device
    B, S = src_ids.shape


    enc_mask = build_encoder_mask(src_ids, pad_id=pad_id)  # [B,1,1,S]
    src_hidden = model.embedding(src_ids)                  # [B,S,C]
    for layer in model.encoder:
        src_hidden = layer(src_hidden, enc_mask)
    memory = src_hidden


    out = torch.full((B, 1), pad_id, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        dec_mask = build_decoder_mask(out, pad_id=pad_id)         # [B,1,T,T]
        tgt_hidden = model.embedding(out)                          # [B,T,C]
        for layer in model.decoder:
            tgt_hidden = layer(tgt_hidden, memory, enc_mask, dec_mask)
        logits = model.head(tgt_hidden)                            # [B,T,V]
        next_token = logits[:, -1].argmax(-1, keepdim=True)        # [B,1]
        out = torch.cat([out, next_token], dim=1)
        if (next_token == eos_id).all():
            break

    return out


# ----------------------------
# 6) 单句翻译
# ----------------------------
def translate_one(model, tokenizer, zh_text: str, max_src_len: int, max_tgt_len: int) -> str:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    assert pad_id is not None and eos_id is not None, "tokenizer 缺少 pad/eos"

  
    src_ids = tokenizer.encode(zh_text, add_special_tokens=False)
    src_ids = src_ids[: max_src_len - 1] + [eos_id]
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE)[None, :]  # [1,S]

    out = greedy_decode(model, src, pad_id=pad_id, eos_id=eos_id, max_len=max_tgt_len)

    pred = out[0, 1:].tolist()

    if eos_id in pred:
        pred = pred[: pred.index(eos_id) + 1]
    return tokenizer.decode(pred, skip_special_tokens=True)


# ----------------------------
# 7) main
# ----------------------------
def main():
    args = get_args()
    model = build_module(args.config, args.weight_path)
    tokenizer = build_tokenizer(args.tokenizer_path)

    en = translate_one(model, tokenizer, args.text, args.max_src_len, args.max_tgt_len)
    print("result:",en)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
