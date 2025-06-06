# preprocessing.py (修复标签问题)
import numpy as np
import pandas as pd
import dgl
import torch
import scipy.sparse as sp
import random

# 配置路径
relation2id_path = '../data/relation2id.csv'
kg_path = '../data/kg_triplet.csv'
slkb_path = '../data/SLKB_rawSL.csv'
dbid2name_path = '../data/dbid2name.csv'
kg_save = '../data/kg2id.txt'
sl_save = '../data/sl2id.txt'
entity_save = '../data/entity2id.txt'

# -----------------------------------------------------Begin-------------------------------------------
# 步骤1: 创建基因名称到实体ID的映射
print('Creating gene name to entity ID mapping...')
dbid2name = pd.read_csv(dbid2name_path)
name_to_id = pd.Series(dbid2name['_id'].values, index=dbid2name['name']).to_dict()

# 步骤2: 加载SLKB数据并筛选RPE1细胞系
print('Loading SLKB data and filtering for RPE1 cell line...')
slkb_data = pd.read_csv(slkb_path)
human_SL = slkb_data[slkb_data['cell_line_origin'] == 'RPE1']
#human_SL = slkb_data
# 映射基因名称到实体ID
human_SL['gene_1_id'] = human_SL['gene_1'].map(name_to_id)
human_SL['gene_2_id'] = human_SL['gene_2'].map(name_to_id)

# 删除无法映射的基因对
original_count = len(human_SL)
human_SL = human_SL.dropna(subset=['gene_1_id', 'gene_2_id'])
human_SL = human_SL.astype({'gene_1_id': int, 'gene_2_id': int})
print(f'Filtered out {original_count - len(human_SL)} pairs with unmapped genes')

# 创建最终数据框
human_SL = human_SL.rename(columns={
    'gene_1_id': 'gene_a.identifier',
    'gene_2_id': 'gene_b.identifier'
})

# 映射标签
human_SL.loc[human_SL['SL_or_not'] == 'Not SL', 'type'] = 0
human_SL.loc[human_SL['SL_or_not'] == 'SL', 'type'] = 1
human_SL = human_SL[['gene_a.identifier', 'gene_b.identifier', 'type']]
human_SL['type'] = human_SL['type'].astype(int)

# 重置索引
human_SL = human_SL.reset_index(drop=True)

# 确保正负样本比例为1:1
def balance_positive_negative_samples(df):
    """确保正负样本比例为1:1"""
    # 分离正负样本
    positive_samples = df[df['type'] == 1]
    negative_samples = df[df['type'] == 0]
    
    # 统计数量
    pos_count = len(positive_samples)
    neg_count = len(negative_samples)
    
    print(f'Original counts: Positive={pos_count}, Negative={neg_count}')
    
    # 确保有足够的样本
    if pos_count == 0 or neg_count == 0:
        raise ValueError("Cannot balance dataset: one class has no samples")
    
    # 如果正样本多于负样本，随机抽取负样本使其匹配
    if pos_count > neg_count:
        # 复制负样本直到达到正样本数量
        repeat_count = (pos_count // neg_count) + 1
        repeated_negatives = pd.concat([negative_samples] * repeat_count, ignore_index=True)
        
        # 随机抽样到与正样本相同数量
        balanced_negatives = repeated_negatives.sample(n=pos_count, random_state=42)
        df = pd.concat([positive_samples, balanced_negatives], ignore_index=True)
        
        print(f'Balanced by repeating negatives: New counts - Positive={pos_count}, Negative={pos_count}')
    
    # 如果负样本多于正样本，随机抽取正样本使其匹配
    elif neg_count > pos_count:
        # 复制正样本直到达到负样本数量
        repeat_count = (neg_count // pos_count) + 1
        repeated_positives = pd.concat([positive_samples] * repeat_count, ignore_index=True)
        
        # 随机抽样到与负样本相同数量
        balanced_positives = repeated_positives.sample(n=neg_count, random_state=42)
        
        # 修复：正确合并平衡后的正样本和原始负样本
        df = pd.concat([balanced_positives, negative_samples], ignore_index=True)
        
        print(f'Balanced by repeating positives: New counts - Positive={neg_count}, Negative={neg_count}')
    
    # 如果已经平衡，直接返回
    else:
        print('Dataset is already balanced with 1:1 ratio')
    
    # 随机打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# 确保正负样本比例为1:1
human_SL = balance_positive_negative_samples(human_SL)
print(f'Final human_SL pairs count: {len(human_SL)}')

# 步骤3: 加载知识图谱
print('Loading knowledge graph...')
relation2id = pd.read_csv(relation2id_path)
kg = pd.read_csv(kg_path, sep=',')

# 映射关系名称到ID
for i in range(len(relation2id)):
    a = relation2id['type'][i]
    b = relation2id['id'][i]
    kg.loc[kg['type(r)'] == a, 'type(r)'] = b

# 删除特定关系 (SL, nonSL, SR)
kg_delete = kg[(kg['type(r)'] != 1) & (kg['type(r)'] != 14) & (kg['type(r)'] != 24)]

# 调整列顺序
order = ['ID(a)', 'type(r)', 'ID(b)']
kg_delete = kg_delete[order].reset_index(drop=True)

# 重新索引关系
relation_old = list(set(kg_delete['type(r)']))
relation_map = {old_id: new_id for new_id, old_id in enumerate(relation_old)}
kg_delete['type(r)'] = kg_delete['type(r)'].map(relation_map)

print('First 10 kg triples:')
print(kg_delete.head(10))

# 步骤4: 删除与SL基因对相同的KG三元组
print('Removing kg triples that match SL pairs...')
index_list = []
for _, row in human_SL.iterrows():
    gene_a = row['gene_a.identifier']
    gene_b = row['gene_b.identifier']
    
    # 查找匹配的三元组
    direct_match = kg_delete[(kg_delete['ID(a)'] == gene_a) & (kg_delete['ID(b)'] == gene_b)]
    reverse_match = kg_delete[(kg_delete['ID(b)'] == gene_a) & (kg_delete['ID(a)'] == gene_b)]
    
    index_list.extend(direct_match.index.tolist())
    index_list.extend(reverse_match.index.tolist())

# 删除重复索引并去重
list_same = list(set(index_list))
print(f'Found {len(list_same)} matching kg triples to remove')

kg_delete = kg_delete.drop(list_same).reset_index(drop=True)

# 步骤5: 删除不在KG中的基因
print('Filtering genes not in knowledge graph...')
kg_genes = set(kg_delete['ID(a)']) | set(kg_delete['ID(b)'])
sl_genes = set(human_SL['gene_a.identifier']) | set(human_SL['gene_b.identifier'])

# 找出不在KG中的基因
missing_genes = sl_genes - kg_genes
print(f'Found {len(missing_genes)} genes in SL pairs not in kg')

# 删除包含这些基因的行
if missing_genes:
    mask = human_SL.apply(lambda row: 
                          row['gene_a.identifier'] in missing_genes or 
                          row['gene_b.identifier'] in missing_genes, axis=1)
    human_SL = human_SL[~mask]

print(f'Human_SL after filtering: {len(human_SL)} pairs')
human_SL = human_SL.reset_index(drop=True)

# 再次平衡数据（因为可能删除了部分样本）
human_SL = balance_positive_negative_samples(human_SL)
print(f'Final balanced human_SL pairs count: {len(human_SL)}')

# 步骤6: 创建全局实体映射
print('Creating global entity mapping...')
all_entities = sorted(set(kg_delete['ID(a)']) | set(kg_delete['ID(b)']) | 
                   set(human_SL['gene_a.identifier']) | set(human_SL['gene_b.identifier']))
entity_map = {entity: idx for idx, entity in enumerate(all_entities)}
print(f'Total entities: {len(all_entities)}')

# 应用全局映射
kg_delete['ID(a)'] = kg_delete['ID(a)'].map(entity_map)
kg_delete['ID(b)'] = kg_delete['ID(b)'].map(entity_map)
human_SL['gene_a.identifier'] = human_SL['gene_a.identifier'].map(entity_map)
human_SL['gene_b.identifier'] = human_SL['gene_b.identifier'].map(entity_map)

# 步骤7: 保存结果
print('Saving results...')
print('First 10 kg triples after mapping:')
print(kg_delete.head(10))
print(f'Final kg triples count: {len(kg_delete)}')

print('First 10 SL pairs after mapping:')
print(human_SL.head(10))
print(f'Final SL pairs count: {len(human_SL)}')

# 检查最终正负样本比例
pos_count = human_SL[human_SL['type'] == 1].shape[0]
neg_count = human_SL[human_SL['type'] == 0].shape[0]
print(f'Final ratio: Positive={pos_count}, Negative={neg_count} (Ratio={pos_count/(pos_count+neg_count):.2f}:{neg_count/(pos_count+neg_count):.2f})')

# 创建实体ID映射文件
entity2id = pd.DataFrame({
    'a': list(entity_map.keys()),
    'b': list(entity_map.values())
})

# 保存文件
kg_delete.to_csv(kg_save, index=False, header=None, sep='\t')
human_SL.to_csv(sl_save, index=False, header=None, sep='\t')
entity2id.to_csv(entity_save, index=False, sep='\t')

# 验证保存的文件
print('Verifying saved files...')
sl2id_df = pd.read_csv(sl_save, sep='\t', header=None, names=['gene_a', 'gene_b', 'label'])
print(f'SL2ID file label distribution: {sl2id_df["label"].value_counts()}')
print('Sample labels:')
print(sl2id_df.head(10))

print('Preprocessing completed successfully!')