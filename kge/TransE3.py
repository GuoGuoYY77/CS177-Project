#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_transe.py
Usage:
    python train_transe.py \
        --train kg_train_triples.txt \
        --valid kg_valid_triples.txt \
        --entity2id kg_entity2id.json \
        --relation2id kg_relation2id.json \
        --dim 128 --batch_size 1024 --epochs 100
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------
# Dataset
# -----------------------------------------------------------

class KGDataset(Dataset):
    """三元组数据集，附带简单的负采样"""

    def __init__(self,
                 triples_file: str,
                 num_entities: int,
                 num_relations: int,
                 negative_sampling_rate: int = 10):
        self.triples = self._load_triples(triples_file)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.neg_rate = negative_sampling_rate
        self._validate()

    @staticmethod
    def _load_triples(file_path: str) -> List[Tuple[int, int, int]]:
        triples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                h, r, t = map(int, line.strip().split())
                triples.append((h, r, t))
        return triples

    def _validate(self) -> None:
        hs, rs, ts = zip(*self.triples)
        min_ent, max_ent = min(min(hs), min(ts)), max(max(hs), max(ts))
        min_rel, max_rel = min(rs), max(rs)

        print("数据验证:")
        print(
            f"  实体ID范围: [{min_ent}, {max_ent}], 期望: [0, {self.num_entities-1}]")
        print(
            f"  关系ID范围: [{min_rel}, {max_rel}], 期望: [0, {self.num_relations-1}]")

        if not (0 <= min_ent and max_ent < self.num_entities):
            raise ValueError("实体 ID 超范围")
        if not (0 <= min_rel and max_rel < self.num_relations):
            raise ValueError("关系 ID 超范围")

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, Tuple]:
        h, r, t = self.triples[idx]
        negatives = []
        for _ in range(self.neg_rate):
            if random.random() < 0.5:            # 替换头实体
                neg_h = random.randint(0, self.num_entities - 1)
                negatives.append((neg_h, r, t))
            else:                                # 替换尾实体
                neg_t = random.randint(0, self.num_entities - 1)
                negatives.append((h, r, neg_t))
        return {"positive": (h, r, t), "negative": negatives}


# -----------------------------------------------------------
# collate_fn
# -----------------------------------------------------------

def kg_collate(batch):
    """把 list(dict) 转成张量字典"""
    pos_h, pos_r, pos_t = [], [], []
    neg_h, neg_r, neg_t = [], [], []

    for sample in batch:
        h, r, t = sample["positive"]
        pos_h.append(h)
        pos_r.append(r)
        pos_t.append(t)
        for nh, nr, nt in sample["negative"]:
            neg_h.append(nh)
            neg_r.append(nr)
            neg_t.append(nt)

    collated = {
        "pos_h": torch.tensor(pos_h, dtype=torch.long),
        "pos_r": torch.tensor(pos_r, dtype=torch.long),
        "pos_t": torch.tensor(pos_t, dtype=torch.long),
        "neg_h": torch.tensor(neg_h, dtype=torch.long),
        "neg_r": torch.tensor(neg_r, dtype=torch.long),
        "neg_t": torch.tensor(neg_t, dtype=torch.long),
    }
    return collated


# -----------------------------------------------------------
# TransE model
# -----------------------------------------------------------

class TransE(nn.Module):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int = 100,
                 margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self._init_emb()

    def _init_emb(self):
        nn.init.xavier_uniform_(self.entity_emb.weight.data)
        nn.init.xavier_uniform_(self.relation_emb.weight.data)
        self.entity_emb.weight.data = nn.functional.normalize(
            self.entity_emb.weight.data, p=2, dim=1
        )

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        return torch.norm(h + r - t, p=2, dim=1)

    def loss(self, pos_triplet, neg_triplet):
        pos_h, pos_r, pos_t = pos_triplet
        neg_h, neg_r, neg_t = neg_triplet
        pos_score = self.forward(pos_h, pos_r, pos_t)
        neg_score = self.forward(neg_h, neg_r, neg_t)
        return torch.relu(self.margin + pos_score - neg_score).mean()


# -----------------------------------------------------------
# Train / Eval helpers
# -----------------------------------------------------------

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        loss = model.loss(
            (batch["pos_h"], batch["pos_r"], batch["pos_t"]),
            (batch["neg_h"], batch["neg_r"], batch["neg_t"])
        )
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main(args):
    # ---------- 加载映射 ----------
    entity2id = json.loads(Path(args.entity2id).read_text(encoding="utf-8"))
    relation2id = json.loads(
        Path(args.relation2id).read_text(encoding="utf-8"))

    num_entities = max(map(int, entity2id.values())) + 1
    num_relations = max(map(int, relation2id.values())) + 1
    print(f"实体总数: {num_entities}, 关系总数: {num_relations}")

    # ---------- 数据集 ----------
    train_ds = KGDataset(args.train, num_entities, num_relations,
                         negative_sampling_rate=args.neg_rate)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=kg_collate)

    # ---------- 模型 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransE(num_entities, num_relations,
                   embedding_dim=args.dim,
                   margin=args.margin).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---------- 训练 ----------
    best_loss = float("inf")
    patience = 0
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch:03d}] loss = {avg_loss:.4f}")

        # 早停
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if args.early_stop and patience >= args.early_stop:
                print("早停触发，结束训练")
                break

        # 周期性保存
        if args.save_every and epoch % args.save_every == 0:
            torch.save(model.state_dict(), f"transe_epoch{epoch}.pth")

    # ---------- 保存最终嵌入 ----------
    entity_emb = model.entity_emb.weight.detach().cpu().numpy()
    relation_emb = model.relation_emb.weight.detach().cpu().numpy()
    np.save("entity_embeddings.npy", entity_emb)
    np.save("relation_embeddings.npy", relation_emb)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    Path("id2entity.json").write_text(json.dumps(id2entity, indent=2,
                                                 ensure_ascii=False), encoding="utf-8")
    Path("id2relation.json").write_text(json.dumps(id2relation, indent=2,
                                                   ensure_ascii=False), encoding="utf-8")
    print("嵌入已保存！")


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TransE on a KG")
    parser.add_argument("--train", required=True, help="训练集三元组文件")
    parser.add_argument("--valid", help="验证集（暂未用，可留空）")
    parser.add_argument("--entity2id", required=True, help="实体映射 JSON")
    parser.add_argument("--relation2id", required=True, help="关系映射 JSON")

    parser.add_argument("--dim", type=int, default=128, help="嵌入维度")
    parser.add_argument("--margin", type=float, default=1.0, help="margin")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--neg_rate", type=int, default=10, help="负采样倍数")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=0,
                        help="每 N epoch 存一次模型；0 表示不存")
    parser.add_argument("--early_stop", type=int, default=0,
                        help="验证 loss 连续 N epoch 不下降则早停；0 表示不早停")

    main(parser.parse_args())
