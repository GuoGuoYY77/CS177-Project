# Use the KG in KG4SL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from typing import Dict, List, Tuple
import dgl
import scipy.sparse as sp


class PreprocessedKGProcessor:

    def __init__(self):
        self.entity2id = {}
        self.id2entity = {}
        self.kg_triplets = None
        self.sl_data = None
        self.num_entities = 0
        self.num_relations = 0

    def load_preprocessed_data(self, kg_path='./Datasets/kg/kg2id.txt',
                               sl_path='./Datasets/kg/sl2id.txt',
                               entity_path='./Datasets/kg/entity2id.txt'):
        """
        加载你预处理后的数据文件

        Args:
            kg_path: 知识图谱三元组文件路径
            sl_path: 合成致死数据文件路径  
            entity_path: 实体映射文件路径
        """
        print("Loading preprocessed data...")

        # 1. 加载实体映射
        entity_df = pd.read_csv(entity_path, sep='\t', header=0)
        self.entity2id = dict(zip(entity_df.iloc[:, 0], entity_df.iloc[:, 1]))
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.num_entities = len(self.entity2id)

        # 2. 加载知识图谱三元组
        kg_df = pd.read_csv(kg_path, sep='\t', header=None)
        kg_df.columns = ['head', 'relation', 'tail']
        self.kg_triplets = kg_df
        self.num_relations = len(kg_df['relation'].unique())

        # 3. 加载合成致死数据
        sl_df = pd.read_csv(sl_path, sep='\t', header=None)
        sl_df.columns = ['gene_a', 'gene_b', 'label']
        self.sl_data = sl_df

        print(f"Entities: {self.num_entities}")
        print(f"Relations: {self.num_relations}")
        print(f"KG triplets: {len(self.kg_triplets)}")
        print(f"SL pairs: {len(self.sl_data)}")

        return self

    def build_dgl_graph(self):
        """构建DGL图用于GNN训练"""
        # 提取所有边
        heads = torch.tensor(self.kg_triplets['head'].values, dtype=torch.long)
        tails = torch.tensor(self.kg_triplets['tail'].values, dtype=torch.long)
        relations = torch.tensor(
            self.kg_triplets['relation'].values, dtype=torch.long)

        # 创建DGL图（双向）
        u = torch.cat([heads, tails])
        v = torch.cat([tails, heads])
        edge_types = torch.cat([relations, relations])

        g = dgl.graph((u, v), num_nodes=self.num_entities)
        g.edata['rel_type'] = edge_types

        return g

    def build_pytorch_geometric_data(self):
        """构建PyTorch Geometric数据格式"""
        # 构建边索引
        heads = self.kg_triplets['head'].values
        tails = self.kg_triplets['tail'].values
        relations = self.kg_triplets['relation'].values

        # 创建双向边
        edge_index = np.vstack([
            np.hstack([heads, tails]),
            np.hstack([tails, heads])
        ])
        edge_attr = np.hstack([relations, relations])

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        # 节点特征（初始化为one-hot或随机）
        x = torch.eye(self.num_entities)  # 简单的one-hot编码

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data


class GraphEmbeddingModel(nn.Module):
    """图神经网络模型用于生成KG Embedding"""

    def __init__(self, num_entities, num_relations, hidden_dim=128,
                 embedding_dim=128, num_layers=3, model_type='GCN'):
        super(GraphEmbeddingModel, self).__init__()

        self.num_entities = num_entities
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.model_type = model_type

        # 实体嵌入
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        # 关系嵌入
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        # 图神经网络层
        self.convs = nn.ModuleList()
        if model_type == 'GCN':
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif model_type == 'GraphSAGE':
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        elif model_type == 'GAT':
            self.convs.append(
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim, heads=4, concat=False))

        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, entity_ids, edge_index, edge_attr=None):
        # 获取实体嵌入
        x = self.entity_embedding(entity_ids)

        # 通过图神经网络层
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # 输出嵌入
        embeddings = self.output_proj(x)
        return F.normalize(embeddings, p=2, dim=1)


class SyntheticLethalityPredictor:
    """合成致死预测器 - 适配你的预处理数据"""

    def __init__(self, kg_processor: PreprocessedKGProcessor):
        self.kg_processor = kg_processor
        self.model = None
        self.embeddings = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def train_embeddings(self, epochs=200, lr=0.01, batch_size=None):
        """训练KG Embedding模型"""
        # 确保device正确设置
        if not hasattr(self, 'device'):
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        print(f"使用设备: {self.device}")

        try:
            print("Building graph data...")
            graph_data = self.kg_processor.build_pytorch_geometric_data()
            print(
                f"图数据构建完成: 实体数 = {self.kg_processor.num_entities}, 关系数 = {self.kg_processor.num_relations}")

            # 创建模型
            self.model = GraphEmbeddingModel(
                num_entities=self.kg_processor.num_entities,
                num_relations=self.kg_processor.num_relations,
                hidden_dim=128,
                embedding_dim=256,
                model_type='GCN'
            ).to(self.device)

            # 确认模型是否在GPU上
            print(f"模型是否在正确设备上: {next(self.model.parameters()).device}")

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            # 移动数据到设备
            entity_ids = torch.arange(
                self.kg_processor.num_entities).to(self.device)
            edge_index = graph_data.edge_index.to(self.device)
            edge_attr = graph_data.edge_attr.to(self.device)

            # 确认数据是否在正确设备上
            print(
                f"数据是否在正确设备上: 实体={entity_ids.device}, 边索引={edge_index.device}")

            print("Starting KG Embedding training...")

            # 显示初始GPU内存使用
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                print(
                    f'初始GPU内存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

            for epoch in range(epochs):
                print(f"开始第 {epoch+1}/{epochs} 个epoch")
                self.model.train()
                optimizer.zero_grad()

                # 前向传播
                try:
                    embeddings = self.model(entity_ids, edge_index, edge_attr)
                    print(f"前向传播成功, 嵌入形状: {embeddings.shape}")
                except Exception as e:
                    print(f"前向传播失败: {e}")
                    import traceback
                    traceback.print_exc()
                    break

                pos_edges = edge_index.t()
                print(f"正样本边数量: {pos_edges.size(0)}")

                # 正样本分数
                try:
                    pos_scores = torch.sum(
                        embeddings[pos_edges[:, 0]] * embeddings[pos_edges[:, 1]], dim=1)
                except Exception as e:
                    print(f"计算正样本分数失败: {e}")
                    traceback.print_exc()
                    break

                # 负采样
                try:
                    neg_edges = self.negative_sampling(edge_index, self.kg_processor.num_entities,
                                                       pos_edges.size(0)).to(self.device)
                    print(f"负采样完成, 形状: {neg_edges.shape}")
                    neg_scores = torch.sum(
                        embeddings[neg_edges[:, 0]] * embeddings[neg_edges[:, 1]], dim=1)
                except Exception as e:
                    print(f"负采样或计算负样本分数失败: {e}")
                    traceback.print_exc()
                    break

                try:
                    pos_loss = - \
                        torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
                    neg_loss = - \
                        torch.log(1 - torch.sigmoid(neg_scores) + 1e-8).mean()
                    loss = pos_loss + neg_loss
                except Exception as e:
                    print(f"计算损失失败: {e}")
                    traceback.print_exc()
                    break

                # 反向传播
                try:
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(f"反向传播或优化器步骤失败: {e}")
                    traceback.print_exc()
                    break

                if epoch % 10 == 0:
                    print(
                        f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Pos Loss: {pos_loss.item():.4f}, Neg Loss: {neg_loss.item():.4f}')

                    # 监控GPU内存
                    if torch.cuda.is_available():
                        print(
                            f'GPU内存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
                        print(
                            f'GPU内存峰值: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')

                print(f"完成第 {epoch+1}/{epochs} 个epoch")
            print("训练完成，生成最终嵌入")
            self.model.eval()
            with torch.no_grad():
                try:
                    self.embeddings = self.model(
                        entity_ids, edge_index, edge_attr).cpu().numpy()
                    print(f"嵌入生成完成，形状: {self.embeddings.shape}")
                except Exception as e:
                    print(f"生成最终嵌入失败: {e}")
                    traceback.print_exc()

            print("KG Embedding training completed!")
            return self.embeddings

        except Exception as e:
            print(f"训练过程中出现未捕获的错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def negative_sampling(self, edge_index, num_entities, num_samples):
        """负采样"""
        edge_set = set(map(tuple, edge_index.t().cpu().numpy()))

        neg_edges = []
        while len(neg_edges) < num_samples:
            i = torch.randint(0, num_entities, (1,)).item()
            j = torch.randint(0, num_entities, (1,)).item()
            if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
                neg_edges.append([i, j])

        return torch.tensor(neg_edges, dtype=torch.long, device=self.device)

    def predict_synthetic_lethality(self, test_ratio=0.2, classifier_type='rf'):

        if self.embeddings is None:
            raise ValueError("Please train KG Embedding model first")

        print("Preparing SL prediction data...")

        features = []
        labels = []

        for _, row in self.kg_processor.sl_data.iterrows():
            gene_a_id = int(row['gene_a'])
            gene_b_id = int(row['gene_b'])
            label = int(row['label'])
            emb_a = self.embeddings[gene_a_id]
            emb_b = self.embeddings[gene_b_id]

            feature = np.concatenate([
                emb_a,
                emb_b,
                emb_a * emb_b,
                np.abs(emb_a - emb_b),
                [np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a)
                                         * np.linalg.norm(emb_b))]
            ])

            features.append(feature)
            labels.append(label)

        features = np.array(features)
        labels = np.array(labels)

        print(f"Feature shape: {features.shape}")
        print(f"Positive samples: {np.sum(labels == 1)}")
        print(f"Negative samples: {np.sum(labels == 0)}")

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_ratio, random_state=42, stratify=labels
        )

        if classifier_type == 'rf':
            classifier = RandomForestClassifier(
                n_estimators=100, random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression(random_state=42)

        print("Training classifier...")
        classifier.fit(X_train, y_train)

        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        y_pred = classifier.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n=== Synthetic Lethality Prediction Results ===")
        print(f"AUC: {auc:.4f}")
        print(f"AP: {ap:.4f}")
        print(f"Accuracy: {acc:.4f}")

        return {
            'classifier': classifier,
            'test_auc': auc,
            'test_ap': ap,
            'test_accuracy': acc,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def save_embeddings(self, filepath: str):
        """保存嵌入向量和相关数据"""
        embedding_data = {
            'embeddings': self.embeddings,
            'entity2id': self.kg_processor.entity2id,
            'id2entity': self.kg_processor.id2entity,
            'num_entities': self.kg_processor.num_entities,
            'num_relations': self.kg_processor.num_relations
        }

        with open(filepath, 'wb') as f:
            pickle.dump(embedding_data, f)
        print(f"KG Embedding saved to: {filepath}")

    def load_embeddings(self, filepath: str):
        """加载嵌入向量和相关数据"""
        with open(filepath, 'rb') as f:
            embedding_data = pickle.load(f)

        self.embeddings = embedding_data['embeddings']
        self.kg_processor.entity2id = embedding_data['entity2id']
        self.kg_processor.id2entity = embedding_data['id2entity']
        self.kg_processor.num_entities = embedding_data['num_entities']
        self.kg_processor.num_relations = embedding_data['num_relations']
        print(f"KG Embedding loaded from {filepath}")


def main():
    """主函数 - 使用你预处理的数据"""

    kg_processor = PreprocessedKGProcessor()

    kg_processor.load_preprocessed_data(
        kg_path='./Datasets/kg/kg2id.txt',
        sl_path='./Datasets/kg/sl2id.txt',
        entity_path='./Datasets/kg/entity2id.txt'
    )

    sl_predictor = SyntheticLethalityPredictor(kg_processor)

    print("\n=== Training KG Embeddings ===")
    embeddings = sl_predictor.train_embeddings(epochs=200, lr=0.01)

    sl_predictor.save_embeddings('./Datasets/kg/kg_embeddings.pkl')


if __name__ == "__main__":
    main()


def analyze_embeddings(kg_processor, embeddings, save_path='./analysis/'):
    """分析嵌入向量的质量"""
    import os
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    plt.title('KG Embeddings PCA Visualization')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(os.path.join(save_path, 'embeddings_pca.png'))
    plt.close()

    stats = {
        'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
        'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
        'mean_cosine_sim': np.mean([np.dot(embeddings[i], embeddings[j])
                                   for i in range(min(100, len(embeddings)))
                                   for j in range(i+1, min(100, len(embeddings)))])
    }

    print("Embedding Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

    return stats
