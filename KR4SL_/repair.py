import json


def check_data_consistency():
    # 加载映射文件
    with open('kg_entity2id.json', 'r') as f:
        entity2id = json.load(f)
    with open('kg_relation2id.json', 'r') as f:
        relation2id = json.load(f)

    print(f"映射文件中的实体数量: {len(entity2id)}")
    print(f"映射文件中的关系数量: {len(relation2id)}")
    print(f"实体ID范围: 0 - {len(entity2id)-1}")
    print(f"关系ID范围: 0 - {len(relation2id)-1}")

    # 检查训练数据
    entity_ids_in_data = set()
    relation_ids_in_data = set()

    with open('kg_train_triples.txt', 'r') as f:
        for i, line in enumerate(f):
            if i < 10:  # 只打印前10行作为示例
                print(f"训练数据示例: {line.strip()}")

            h, r, t = map(int, line.strip().split())
            entity_ids_in_data.update([h, t])
            relation_ids_in_data.add(r)

    print(
        f"\n训练数据中的实体ID范围: {min(entity_ids_in_data)} - {max(entity_ids_in_data)}")
    print(
        f"训练数据中的关系ID范围: {min(relation_ids_in_data)} - {max(relation_ids_in_data)}")
    print(f"训练数据中的关系ID数量: {len(relation_ids_in_data)}")

    # 检查是否有超出范围的ID
    max_entity_in_mapping = len(entity2id) - 1
    max_relation_in_mapping = len(relation2id) - 1

    invalid_entities = [
        eid for eid in entity_ids_in_data if eid > max_entity_in_mapping]
    invalid_relations = [
        rid for rid in relation_ids_in_data if rid > max_relation_in_mapping]

    if invalid_entities:
        print(f"\n发现超出范围的实体ID: {invalid_entities[:10]}... (显示前10个)")
    if invalid_relations:
        print(f"发现超出范围的关系ID: {invalid_relations[:10]}... (显示前10个)")

    return len(entity2id), len(relation2id)


# 运行检查
check_data_consistency()


def regenerate_training_data():
    """重新生成训练数据，确保使用正确的映射ID"""

    # 加载映射文件
    with open('kg_entity2id.json', 'r') as f:
        entity2id = json.load(f)
    with open('kg_relation2id.json', 'r') as f:
        relation2id = json.load(f)

    # 重新处理原始kg.txt文件
    mapped_triples = []
    skipped_count = 0

    with open('F:\Courses\大三下\生物信息学\SL_Project_Baseline\KR4SL\data\kg.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' is_a ')
                if len(parts) == 2:
                    head, tail = parts[0], parts[1]
                    relation = 'is_a'

                    # 检查实体和关系是否在映射中
                    if head in entity2id and tail in entity2id and relation in relation2id:
                        head_id = entity2id[head]
                        relation_id = relation2id[relation]
                        tail_id = entity2id[tail]
                        mapped_triples.append((head_id, relation_id, tail_id))
                    else:
                        skipped_count += 1

    print(f"成功映射的三元组数量: {len(mapped_triples)}")
    print(f"跳过的三元组数量: {skipped_count}")

    # 划分训练集和验证集
    import random
    random.shuffle(mapped_triples)
    split_idx = int(0.8 * len(mapped_triples))

    train_triples = mapped_triples[:split_idx]
    valid_triples = mapped_triples[split_idx:]

    # 保存新的训练数据
    with open('kg_train_triples_fixed.txt', 'w') as f:
        for h, r, t in train_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    with open('kg_valid_triples_fixed.txt', 'w') as f:
        for h, r, t in valid_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    print("已生成修复后的训练数据文件:")
    print("- kg_train_triples_fixed.txt")
    print("- kg_valid_triples_fixed.txt")


# 重新生成数据
regenerate_training_data()
