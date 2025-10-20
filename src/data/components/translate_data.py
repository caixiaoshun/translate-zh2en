import json
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TranslateDataset(Dataset):
    def __init__(
        self,
        json_path: str = "data/translation2019zh_train.json",
        tokenizer_name: str = "checkpoints/mt5-small",
        max_src_len: int = 128,
        max_tgt_len: int = 128,
    ):
        self.samples: List[Dict[str, str]] = []
        with open(json_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.samples.append(
                    {"src": obj["chinese"].strip(), "tgt": obj["english"].strip()}
                )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.eos_id = self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, text: str, max_len: int) -> List[int]:
        ids: List[int] = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        
        ids = ids[: max_len - 1] 
        
        if len(ids) == 0 or ids[-1] != self.eos_id:
            ids = ids + [self.eos_id]
       
        ids = ids[:max_len]
        return ids

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        src_text = item["src"]
        tgt_text = item["tgt"]

        src_ids = self._encode(src_text, self.max_src_len)
        tgt_ids = self._encode(tgt_text, self.max_tgt_len)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }
    
if __name__ == "__main__":
    dataset = TranslateDataset()