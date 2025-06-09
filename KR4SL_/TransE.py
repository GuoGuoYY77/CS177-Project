import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random


class KGDataset(Dataset):
    def __init__(self, triples_file, num_entities, num_relations, negative_sampling_rate=10):
        self.triples = self.load_triples(triples_file)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.negative_sampling_rate = negative_sampling_rate

    def load_triples(self, file_path):
        triples = []
        with open(file_path, 'r') as f:
            for line in f:
                h, r, t = map(int, line.strip().split())
                triples.append((h, r, t))
        return triples

    def __len__(self):
        return len(self.triples)

    def validate_data(self):
        """验证数据范围是否正确"""
        max_entity = max([max(h, t) for h, r, t in self.triples])
        max_relation = max([r for h, r, t in self.triples])
        min_entity = min([min(h, t) for h, r, t in self.triples])
        min_relation = min([r for h, r, t in self.triples])

        print(f"数据验证:")
        print(
            f"实体ID范围: [{min_entity}, {max_entity}], 期望范围: [0, {self.num_entities-1}]")
        print(
            f"关系ID范围: [{min_relation}, {max_relation}], 期望范围: [0, {self.num_relations-1}]")

        if max_entity >= self.num_entities:
            raise ValueError(
                f"实体ID {max_entity} 超出范围 [0, {self.num_entities-1}]")
        if max_relation >= self.num_relations:
            raise ValueError(
                f"关系ID {max_relation} 超出范围 [0, {self.num_relations-1}]")
        if min_entity < 0 or min_relation < 0:
            raise ValueError("发现负数ID")

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        negative_samples = []
        for _ in range(self.negative_sampling_rate):
            if random.random() < 0.5:
                neg_h = random.randint(0, self.num_entities - 1)
                negative_samples.append((neg_h, r, t))
            else:
                neg_t = random.randint(0, self.num_entities - 1)
                negative_samples.append((h, r, neg_t))
        return {
            'positive': (h, r, t),
            'negative': negative_samples
        }


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.init_embeddings()

    def init_embeddings(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        self.entity_embeddings.weight.data = nn.functional.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1)

    def forward(self, heads, relations, tails):
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        scores = torch.norm(h + r - t, p=2, dim=1)
        return scores

    def loss(self, positive_triples, negative_triples):
        pos_heads, pos_rels, pos_tails = positive_triples
        neg_heads, neg_rels, neg_tails = negative_triples
        pos_scores = self.forward(pos_heads, pos_rels, pos_tails)
        neg_scores = self.forward(neg_heads, neg_rels, neg_tails)

        # Margin ranking loss
        loss = torch.relu(self.margin + pos_scores - neg_scores).mean()
        return loss


def train_kg_embedding():
    # 加载映射文件
    with open('kg_entity2id.json', 'r') as f:
        entity2id = json.load(f)
    with open('kg_relation2id.json', 'r') as f:
        relation2id = json.load(f)

    num_entities = max(map(int, entity2id.values())) + 1
    num_relations = max(map(int, relation2id.values())) + 1

    print(f"实体数量: {num_entities}")
    print(f"关系数量: {num_relations}")

    # 创建数据集
    train_dataset = KGDataset(
        'kg_train_triples.txt', num_entities, num_relations)
    valid_dataset = KGDataset(
        'kg_valid_triples.txt', num_entities, num_relations)
    train_dataset.validate_data()
    valid_dataset.validate_data()
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # 初始化模型
    model = TransE(num_entities, num_relations, embedding_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # 准备正样本
            pos_triples = batch['positive']
            pos_heads = torch.tensor([t[0] for t in pos_triples])
            pos_rels = torch.tensor([t[1] for t in pos_triples])
            pos_tails = torch.tensor([t[2] for t in pos_triples])

            # 准备负样本
            neg_samples = batch['negative']
            neg_heads = torch.tensor(
                [sample[0] for samples in neg_samples for sample in samples])
            neg_rels = torch.tensor(
                [sample[1] for samples in neg_samples for sample in samples])
            neg_tails = torch.tensor(
                [sample[2] for samples in neg_samples for sample in samples])

            # 计算损失
            loss = model.loss(
                (pos_heads, pos_rels, pos_tails),
                (neg_heads, neg_rels, neg_tails)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(
                f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')

    return model, entity2id, relation2id


def save_embeddings(model, entity2id, relation2id):
    # 获取嵌入矩阵
    entity_embeddings = model.entity_embeddings.weight.detach().numpy()
    relation_embeddings = model.relation_embeddings.weight.detach().numpy()

    # 保存
    np.save('entity_embeddings.npy', entity_embeddings)
    np.save('relation_embeddings.npy', relation_embeddings)

    # 创建反向映射便于后续使用
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    with open('id2entity.json', 'w') as f:
        json.dump(id2entity, f, indent=2)
    with open('id2relation.json', 'w') as f:
        json.dump(id2relation, f, indent=2)

    print("嵌入已保存!")
    print(f"实体嵌入形状: {entity_embeddings.shape}")
    print(f"关系嵌入形状: {relation_embeddings.shape}")

    return entity_embeddings, relation_embeddings


# 主函数
if __name__ == "__main__":
    # 训练模型
    model, entity2id, relation2id = train_kg_embedding()
    # 保存嵌入
    entity_emb, relation_emb = save_embeddings(model, entity2id, relation2id)

    print("KG嵌入训练完成!")
