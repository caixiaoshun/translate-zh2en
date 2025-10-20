from typing import Any, Dict, Optional, List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.data.components.translate_data import TranslateDataset


class TranslateDataModule(LightningDataModule):

    def __init__(
        self,
        train_path: str = "data/translation2019zh_train.json",
        val_path: str = "data/translation2019zh_valid.json",
        tokenizer_path: str = "checkpoints/mt5-small",
        max_src_len=128,
        max_tgt_len=128,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        pad_id=0,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        TranslateDataset(
            json_path=self.hparams.train_path,
            tokenizer_name=self.hparams.tokenizer_path,
            max_src_len=self.hparams.max_src_len,
            max_tgt_len=self.hparams.max_tgt_len,
        )
        TranslateDataset(
            json_path=self.hparams.val_path,
            tokenizer_name=self.hparams.tokenizer_path,
            max_src_len=self.hparams.max_src_len,
            max_tgt_len=self.hparams.max_tgt_len,
        )


    def pad_1d(self, seqs: List[torch.Tensor], pad_id: int) -> torch.Tensor:
        max_len = max(x.size(0) for x in seqs)
        out = torch.full((len(seqs), max_len), pad_id, dtype=seqs[0].dtype)
        for i, x in enumerate(seqs):
            out[i, : x.size(0)] = x
        return out

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        src_list = [x["src_ids"] for x in batch]
        tgt_list = [x["tgt_ids"] for x in batch]

        src_ids = self.pad_1d(src_list, self.hparams.pad_id)   # [B,S]
        tgt_ids = self.pad_1d(tgt_list, self.hparams.pad_id)   # [B,T]

        B, S = src_ids.size(0), src_ids.size(1)
        T = tgt_ids.size(1)

        # ---- decoder input----
        pad_col = torch.full((B, 1), self.hparams.pad_id, dtype=tgt_ids.dtype)
        tgt_in_ids = torch.cat([pad_col, tgt_ids[:, :-1]], dim=1)  # [B,T]

        # ---- labels----
        labels = tgt_ids.clone()
        labels[labels == self.hparams.pad_id] = -100

        device = src_ids.device
        neg_inf = -1e8

        # 编码器 key padding 掩码：[B,1,1,S]，可见=0，pad=-inf
        enc_key_visible = (src_ids != self.hparams.pad_id)[:, None, None, :]  # [B,1,1,S] (bool)
        encoder_mask = torch.where(enc_key_visible, torch.tensor(0.0, device=device), torch.tensor(neg_inf, device=device))

        # 解码器：因果 + key padding
        # 因果：[1,1,T,T]，上三角(不含对角) = -inf，其余=0
        causal = torch.full((T, T), 0.0, device=device)
        causal = causal.masked_fill(torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1), neg_inf)
        causal = causal[None, None, :, :]  # [1,1,T,T]

        # key padding 掩码（针对 decoder 的 key 维度 T）：[B,1,1,T]
        dec_key_visible = (tgt_in_ids != self.hparams.pad_id)[:, None, None, :]  # [B,1,1,T]
        dec_key_mask = torch.where(dec_key_visible, torch.tensor(0.0, device=device), torch.tensor(neg_inf, device=device))

        # 合并得到解码器自注意力掩码：[B,1,T,T]
        decoder_mask = causal + dec_key_mask  # 自动广播到 [B,1,T,T]

        return {
            "src_ids": src_ids,
            "tgt_in_ids": tgt_in_ids,
            "tgt_out_ids": tgt_ids,
            "labels": labels,
            "encoder_mask": encoder_mask,  # [B,1,1,S], float(0/-inf)
            "decoder_mask": decoder_mask,  # [B,1,T,T], float(0/-inf)
        }


    def setup(self, stage: Optional[str] = None) -> None:

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            self.data_train = TranslateDataset(
                json_path=self.hparams.train_path,
                tokenizer_name=self.hparams.tokenizer_path,
                max_src_len=self.hparams.max_src_len,
                max_tgt_len=self.hparams.max_tgt_len,
            )
            self.data_val = TranslateDataset(
                json_path=self.hparams.val_path,
                tokenizer_name=self.hparams.tokenizer_path,
                max_src_len=self.hparams.max_src_len,
                max_tgt_len=self.hparams.max_tgt_len,
            )
            self.data_test = self.data_val

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
    