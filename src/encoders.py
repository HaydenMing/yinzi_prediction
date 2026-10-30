"""可拆卸模块：原文字符编码器，与报价日时序编码器分开。"""
import torch
from torch import nn
from torch.nn import functional as F


class CharacterEncoder(nn.Module):
    def __init__(self, embed_dim, hidden, kind):
        super().__init__()
        if kind not in {"mean", "textcnn", "bigru"} or hidden < 2 or hidden % 2:
            raise ValueError("字符编码器非法或hidden不是正偶数")
        self.kind = kind
        if kind == "mean":
            self.project = nn.Linear(embed_dim, hidden)
        elif kind == "textcnn":
            self.convs = nn.ModuleList(nn.Conv1d(embed_dim, hidden, k) for k in (3, 4, 5))
            self.project = nn.Linear(3 * hidden, hidden)
        else:
            self.rnn = nn.GRU(embed_dim, hidden // 2, batch_first=True, bidirectional=True)
            self.project = nn.Identity()

    def chunk_features(self, x, lengths):
        if x.ndim != 3 or (lengths < 1).any() or (lengths > x.size(1)).any():
            raise ValueError("字符序列必须非空且长度有效")
        valid = torch.arange(x.size(1), device=x.device)[None] < lengths[:, None]
        x = x * valid[:, :, None]
        if self.kind == "mean":
            return x.sum(1) / lengths[:, None]
        if self.kind == "bigru":
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, state = self.rnn(packed)
            return torch.cat([state[-2], state[-1]], -1)
        if x.size(1) < 5:
            x = F.pad(x, (0, 0, 0, 5 - x.size(1)))
        pooled = []
        for conv in self.convs:
            features = F.gelu(conv(x.transpose(1, 2)))
            count = (lengths - conv.kernel_size[0] + 1).clamp(min=1)
            mask = torch.arange(features.size(-1), device=x.device)[None] < count[:, None]
            pooled.append(features.masked_fill(~mask[:, None], float("-inf")).amax(-1))
        return torch.cat(pooled, -1)

    def forward(self, x, lengths):
        return self.project(self.chunk_features(x, lengths))


class CausalBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(hidden, hidden, 3, dilation=dilation)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(.1)

    def forward(self, x):
        update = self.conv(F.pad(x.transpose(1, 2), (self.padding, 0))).transpose(1, 2)
        return self.norm(x + self.drop(F.gelu(update)))


class TemporalEncoder(nn.Module):
    """报价日维度[B,T,D]的编码；与原文TextCNN不同。"""
    def __init__(self, input_dim, hidden, kind):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(input_dim, hidden, batch_first=True)
        elif kind == "tcn":
            self.project = nn.Linear(input_dim, hidden)
            self.blocks = nn.Sequential(*(CausalBlock(hidden, d) for d in (1, 2, 4)))
        else:
            raise ValueError("价格/共享时序编码器只能是gru或tcn")

    def forward(self, x):
        return self.rnn(x)[0] if self.kind == "gru" else self.blocks(self.project(x))
