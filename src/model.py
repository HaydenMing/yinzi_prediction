"""两种融合位置；显式保留编码特征、融合特征及标准化价差输出。"""
import torch
from torch import nn
from torch.nn import functional as F
from data import chunk_ids
from encoders import CharacterEncoder, TemporalEncoder
from fusion import FusionBackbone


def aggregate_chunks(vectors, owners, weights, count, kind):
    result = []
    for i in range(count):
        positions = [j for j, owner in enumerate(owners) if owner == i]
        values = vectors[positions]
        if kind == "textcnn":
            result.append(values.amax(0))
        else:
            w = vectors.new_tensor([weights[j] for j in positions])
            result.append((values * w[:, None]).sum(0) / w.sum())
    return torch.stack(result)


def run_chunks(encoder, blocks, owners, weights, count, chunk_batch):
    vectors = []
    for start in range(0, len(blocks), chunk_batch):
        part = blocks[start:start + chunk_batch]
        lengths = torch.tensor([len(x) for x in part], device=part[0].device)
        width = max(len(x) for x in part)
        padded = torch.stack([F.pad(x, (0, 0, 0, width - len(x))) for x in part])
        vectors.append(encoder.chunk_features(padded, lengths))
    pooled = aggregate_chunks(torch.cat(vectors), owners, weights, count, encoder.kind)
    return encoder.project(pooled)


class DocumentEncoder(nn.Module):
    """完整文字分块编码。TextCNN跨块最大池化在最终投影之前进行。"""
    def __init__(self, vocab_size, embed_dim, hidden, kind, chunk_size, chunk_batch):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = CharacterEncoder(embed_dim, hidden, kind)
        self.kind, self.hidden = kind, hidden
        self.chunk_size, self.chunk_batch = chunk_size, chunk_batch
        self.cache = {}

    def forward(self, snapshot_ids, snapshots, field):
        blocks, owners, weights, present = [], [], [], []
        for sid in snapshot_ids:
            ids = snapshots[sid][field]
            if not ids:
                continue
            owner = len(present)
            present.append(sid)
            key = (sid, field)
            if key not in self.cache:
                self.cache[key] = chunk_ids(ids, self.chunk_size, 4 if self.kind == "textcnn" else 0)
            for chunk in self.cache[key]:
                blocks.append(self.embedding(torch.tensor(chunk, device=self.embedding.weight.device)))
                owners.append(owner)
                weights.append(len(chunk))
        output = self.embedding.weight.new_zeros(len(snapshot_ids), self.hidden)
        if not present:
            return output
        encoded = run_chunks(self.encoder, blocks, owners, weights, len(present), self.chunk_batch)
        position = {sid: i for i, sid in enumerate(snapshot_ids)}
        indices = torch.tensor([position[sid] for sid in present], device=output.device)
        return output.index_copy(0, indices, encoded)


class JointEncoder(nn.Module):
    """价格投影、新闻及因子字符Embedding先拼接，再共享提取特征。"""
    def __init__(self, vocab_size, embed_dim, hidden, kind, temporal, chunk_size, chunk_batch):
        super().__init__()
        self.characters = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.types = nn.Embedding(4, embed_dim)
        self.price_projection = nn.Linear(2, embed_dim)
        self.state_projection = nn.Linear(5, embed_dim)
        self.encoder = CharacterEncoder(embed_dim, hidden, kind)
        self.temporal = TemporalEncoder(hidden, hidden, temporal)
        self.to_views = nn.Linear(hidden, 3 * hidden)
        self.kind, self.chunk_size, self.chunk_batch = kind, chunk_size, chunk_batch
        self.cache = {}

    def forward(self, price, snapshot_ids, snapshots):
        flat = snapshot_ids.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        sids = unique.tolist()
        first = torch.tensor([int((flat == sid).nonzero()[0, 0]) for sid in sids], device=price.device)
        numeric = price.reshape(-1, 2)[first]
        states = price.new_tensor([snapshots[sid]["flags"] for sid in sids])
        price_tokens = self.price_projection(numeric) + self.types.weight[0]
        state_tokens = self.state_projection(states) + self.types.weight[3]
        blocks, owners, weights = [], [], []
        for owner, sid in enumerate(sids):
            if sid not in self.cache:
                news = snapshots[sid]["news_ids"]
                factor = snapshots[sid]["factor_ids"]
                ids = news + ([2] if news and factor else []) + factor
                tags = [1] * len(news) + ([2] if news and factor else []) + [2] * len(factor)
                overlap = 4 if self.kind == "textcnn" else 0
                self.cache[sid] = chunk_ids(list(zip(ids, tags)), self.chunk_size, overlap) or [[]]
            for chunk in self.cache[sid]:
                prefix = torch.stack([price_tokens[owner], state_tokens[owner]])
                if chunk:
                    tokens = torch.tensor([x[0] for x in chunk], device=price.device)
                    tags = torch.tensor([x[1] for x in chunk], device=price.device)
                    text = self.characters(tokens) + self.types(tags)
                    joint_tokens = torch.cat([prefix, text], dim=0)
                else:
                    joint_tokens = prefix
                blocks.append(joint_tokens)
                owners.append(owner)
                weights.append(max(len(chunk), 1))
        joint_features = run_chunks(self.encoder, blocks, owners, weights, len(sids), self.chunk_batch)
        joint_sequence = joint_features[inverse].reshape(price.shape[0], price.shape[1], -1)
        shared_features = self.temporal(joint_sequence)
        mixed_views = list(self.to_views(shared_features).chunk(3, dim=-1))
        return mixed_views, shared_features


class MultimodalModel(nn.Module):
    def __init__(self, name, recipe, snapshots, vocab_size, hidden=32, embed_dim=32, chunk_size=256, chunk_batch=64):
        super().__init__()
        self.recipe, self.snapshots = dict(recipe), snapshots
        self.backbone = FusionBackbone(name, hidden)
        self.hidden = hidden
        if recipe["order"] == "late":
            self.price_encoder = TemporalEncoder(2, hidden, recipe["price"])
            self.news_encoder = DocumentEncoder(vocab_size, embed_dim, hidden, recipe["news"], chunk_size, chunk_batch)
            self.factor_encoder = DocumentEncoder(vocab_size, embed_dim, hidden, recipe["factor"], chunk_size, chunk_batch)
            self.news_with_flags = nn.Linear(hidden + 5, hidden)
            self.factor_with_flags = nn.Linear(hidden + 5, hidden)
            # 原文编码后再处理报价日序列，两个GRU参数独立。
            self.news_temporal = TemporalEncoder(hidden, hidden, recipe.get("news_temporal", "gru"))
            self.factor_temporal = TemporalEncoder(hidden, hidden, recipe.get("factor_temporal", "gru"))
        elif recipe["order"] == "early":
            self.joint_encoder = JointEncoder(vocab_size, embed_dim, hidden, recipe["joint"], recipe["price"], chunk_size, chunk_batch)
        else:
            raise ValueError("未知融合顺序")

    def forward(self, price, snapshot_ids, return_features=False):
        if price.ndim != 3 or price.shape[-1] != 2 or tuple(price.shape[:2]) != tuple(snapshot_ids.shape):
            raise ValueError("价格[B,T,2]和快照ID[B,T]必须对齐")
        if self.recipe["order"] == "late":
            unique, inverse = torch.unique(snapshot_ids.reshape(-1), sorted=True, return_inverse=True)
            ids = unique.tolist()
            news = self.news_encoder(ids, self.snapshots, "news_ids")
            factor = self.factor_encoder(ids, self.snapshots, "factor_ids")
            flags = price.new_tensor([self.snapshots[sid]["flags"] for sid in ids])
            news = self.news_with_flags(torch.cat([news, flags], -1))[inverse].reshape(*price.shape[:2], -1)
            factor = self.factor_with_flags(torch.cat([factor, flags], -1))[inverse].reshape(*price.shape[:2], -1)
            price_features = self.price_encoder(price)
            news_features = self.news_temporal(news)
            factor_features = self.factor_temporal(factor)
            views = [price_features, news_features, factor_features]
            saved = {"price_features": price_features, "news_features": news_features, "factor_features": factor_features}
        else:
            views, shared = self.joint_encoder(price, snapshot_ids, self.snapshots)
            saved = {"joint_features": shared, **{f"mixed_view_{i+1}": v for i, v in enumerate(views)}}
        out = self.backbone(views)
        if return_features:
            out["encoder_features"] = saved
        return out

    def loss(self, out, y, ids):
        return self.backbone.loss(out, y, ids)

    @torch.no_grad()
    def enqueue(self, out, ids):
        self.backbone.enqueue(out, ids)


