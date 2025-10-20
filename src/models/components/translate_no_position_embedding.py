from torch import nn as nn
import torch
from torch.nn import functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        # 初始化可学习的缩放和平移参数
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        output = (
            self.gamma * x_normalized + self.beta
        )  # 形状: (batch_size, seq_len, hidden_size)

        return output



class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim % 2 == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, x: torch.Tensor, B: int, T: int):
        # [B, T, C] -> [B, H, T, Dh]
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        B, T_q, C = q.shape

        T_k = k.shape[1]

        # 线性映射到 Q/K/V
        Q = self.q_proj(q)  # [B, T_q, C]
        K = self.k_proj(k)  # [B, T_k, C]
        V = self.v_proj(v)  # [B, T_k, C]

        # 拆头
        Q = self._shape(Q, B, T_q)  # [B, H, T_q, Dh]
        K = self._shape(K, B, T_k)  # [B, H, T_k, Dh]
        V = self._shape(V, B, T_k)  # [B, H, T_k, Dh]

        # logits: [B, H, T_q, T_k]
        logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attn_mask is not None:

            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None, :, :]

            logits = logits + attn_mask

        attn = F.softmax(logits.float(), dim=-1).to(Q.dtype)
        attn = self.attn_drop(attn)

        # 加权求和
        out = torch.matmul(attn, V)  # [B, H, T_q, Dh]
        out = out.transpose(1, 2).contiguous().view(B, T_q, C)  # [B, T_q, C]

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out


class FFN(nn.Module):
    def __init__(self, dim, drop_rate):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias, attn_drop, proj_drop, ffn_drop):
        super().__init__()
        self.norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm_2 = LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, drop_rate=ffn_drop)

    def forward(self, x, encoder_mask):

        residual = x
        x = self.norm_1(x)
        x = residual + self.attn(x, x, x, attn_mask=encoder_mask)
        x = x + self.ffn(self.norm_2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias, attn_drop, proj_drop, ffn_drop):
        super().__init__()
        self.norm_1 = LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm_2 = LayerNorm(embed_dim)

        self.cross_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm_3 = LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim, drop_rate=ffn_drop)

    def forward(self, x, e_output, encoder_mask, decoder_mask):
        residual = x
        x = self.norm_1(x)
        x = residual + self.self_attn(x, x, x, attn_mask=decoder_mask)
        x = x + self.cross_attn(
            self.norm_2(x), e_output, e_output, attn_mask=encoder_mask
        )
        x = x + self.ffn(self.norm_3(x))
        return x


class TranslateModelNoPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        encoder_layers,
        decoder_layers,
        embed_dim,
        num_heads,
        qkv_bias,
        attn_drop,
        proj_drop,
        ffn_drop,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, src, tgt_input, src_mask, tgt_mask):
        src_hidden = self.embedding(src)   # [B, Ts, C]
        tgt_hidden = self.embedding(tgt_input)   # [B, Tt, C]

        for layer in self.encoder:
            src_hidden = layer(src_hidden, src_mask)
        src_encoder = src_hidden


        for layer in self.decoder:
            tgt_hidden = layer(tgt_hidden, src_encoder, src_mask, tgt_mask)

        logits = self.head(tgt_hidden)
        return logits
