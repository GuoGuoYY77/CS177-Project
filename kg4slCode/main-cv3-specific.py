import os
import sys
import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import ShuffleSplit

np.random.seed(555)
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=64, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=256, help='dimension of user and entity embeddings')
parser.add_argument('--n_hop', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--l2_weight', type=float, default=0.0039, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--trainv1_testv2', type=bool, default=False, help='train_on_v1_test_on_v2')
parser.add_argument('--earlystop_flag', type=bool, default=True, help='whether early stopping')
args = parser.parse_args()

param_name = 'final'
data = load_data(args)

# Load gene pairs data
print('Reading sl2id file for dataset splitting...')
sl2id_file = '../data/sl2id'
if os.path.exists(sl2id_file + '.npy'):
    sl2id_np = np.load(sl2id_file + '.npy')
else:
    sl2id_np = np.loadtxt(sl2id_file + '.txt', dtype=np.int64)
    np.save(sl2id_file + '.npy', sl2id_np)
np.random.shuffle(sl2id_np)

# For version 2 test data
if args.trainv1_testv2:
    version2_notin_version1_file = '../data/version2_notin_version1'
    if os.path.exists(version2_notin_version1_file + '.npy'):
        version2_notin_version1_np = np.load(version2_notin_version1_file + '.npy')
    else:
        version2_notin_version1_np = np.loadtxt(version2_notin_version1_file + '.txt', dtype=np.int64)
        np.save(version2_notin_version1_file + '.npy', version2_notin_version1_np)
    np.random.shuffle(version2_notin_version1_np)

def strict_split_by_genes(sl2id_np, test_size=0.2):
    """Split dataset ensuring test pairs contain no genes from training set"""
    # Get all unique genes
    all_genes = np.unique(np.concatenate([sl2id_np[:, 0], sl2id_np[:, 1]]))
    
    # Split genes into train and test sets
    n_train_genes = int(len(all_genes) * (1 - test_size))
    train_genes = set(np.random.choice(all_genes, size=n_train_genes, replace=False))
    test_genes = set(all_genes) - train_genes
    
    # Get test pairs where both genes are in test_genes
    test_mask = np.array([(pair[0] in test_genes) and (pair[1] in test_genes) for pair in sl2id_np])
    test_pairs = sl2id_np[test_mask]
    
    # Get train pairs where both genes are in train_genes
    train_mask = np.array([(pair[0] in train_genes) and (pair[1] in train_genes) for pair in sl2id_np])
    train_pairs = sl2id_np[train_mask]
    
    # Ensure we have enough test pairs
    min_test_pairs = int(len(sl2id_np) * test_size)
    if len(test_pairs) < min_test_pairs:
        # If not enough test pairs, we need to adjust our gene split
        # This might require multiple attempts to find a good split
        print(f"Warning: Only found {len(test_pairs)} test pairs (needed {min_test_pairs})")
        print("Adjusting gene split to find more test pairs...")
        
        # Try to find a better gene split that produces more test pairs
        best_test_count = len(test_pairs)
        best_split = (train_genes, test_genes, train_pairs, test_pairs)
        
        for _ in range(10):  # Try up to 10 different random splits
            # Try with slightly different train/test gene ratios
            adjusted_test_size = test_size * (1 + np.random.uniform(-0.2, 0.2))
            n_train_genes = int(len(all_genes) * (1 - adjusted_test_size))
            train_genes = set(np.random.choice(all_genes, size=n_train_genes, replace=False))
            test_genes = set(all_genes) - train_genes
            
            test_mask = np.array([(pair[0] in test_genes) and (pair[1] in test_genes) for pair in sl2id_np])
            test_pairs = sl2id_np[test_mask]
            
            train_mask = np.array([(pair[0] in train_genes) and (pair[1] in train_genes) for pair in sl2id_np])
            train_pairs = sl2id_np[train_mask]
            
            if len(test_pairs) > best_test_count:
                best_test_count = len(test_pairs)
                best_split = (train_genes, test_genes, train_pairs, test_pairs)
                if best_test_count >= min_test_pairs:
                    break
        
        train_genes, test_genes, train_pairs, test_pairs = best_split
        print(f"Best split found: {len(test_pairs)} test pairs")
    
    return train_pairs, test_pairs

# Split dataset with strict gene separation
train_data, test_data = strict_split_by_genes(sl2id_np, test_size=0.2)

if args.trainv1_testv2:
    train_data = sl2id_np
    test_data = version2_notin_version1_np

# Verify the split meets our requirements
train_genes = set(np.unique(np.concatenate([train_data[:, 0], train_data[:, 1]])))
test_genes = set(np.unique(np.concatenate([test_data[:, 0], test_data[:, 1]])))
assert len(train_genes & test_genes) == 0, "Training and test sets share genes!"
print(f"Split complete: {len(train_data)} train pairs, {len(test_data)} test pairs")
print(f"Train genes: {len(train_genes)}, Test genes: {len(test_genes)}")

# Prepare data for training
data = list(data)
data.append(train_data)
data.append(test_data)
data = tuple(data)

# Initialize metrics lists
metrics = {
    'train_auc': [], 'train_f1': [], 'train_aupr': [],
    'eval_auc': [], 'eval_f1': [], 'eval_aupr': [],
    'test_auc': [], 'test_f1': [], 'test_aupr': [],
    'loss': []
}

# Train and evaluate
results = train(args, data, param_name + '_1')
metrics['loss'].append(results[0])
metrics['train_auc'].append(results[1])
metrics['train_f1'].append(results[2])
metrics['train_aupr'].append(results[3])
metrics['eval_auc'].append(results[4])
metrics['eval_f1'].append(results[5])
metrics['eval_aupr'].append(results[6])
metrics['test_auc'].append(results[7])
metrics['test_f1'].append(results[8])
metrics['test_aupr'].append(results[9])

# Calculate mean metrics
print('\nFinal Results:')
print('Training -   AUC: %.4f, AUPR: %.4f, F1: %.4f' % (
    np.mean(metrics['train_auc']), np.mean(metrics['train_aupr']), np.mean(metrics['train_f1'])))
print('Validation - AUC: %.4f, AUPR: %.4f, F1: %.4f' % (
    np.mean(metrics['eval_auc']), np.mean(metrics['eval_aupr']), np.mean(metrics['eval_f1'])))
print('Test -       AUC: %.4f, AUPR: %.4f, F1: %.4f' % (
    np.mean(metrics['test_auc']), np.mean(metrics['test_aupr']), np.mean(metrics['test_f1'])))
print('Training loss: %.4f' % np.mean(metrics['loss']))