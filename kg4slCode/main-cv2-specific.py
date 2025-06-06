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
parser.print_help()

param_name = 'final' # set as the parameter name while adjust parameters
args = parser.parse_args()

data = load_data(args)

# reading rating file for split dataset
print('reading sl2id file again for spliting dataset...')
sl2id_file = '../data/sl2id'
if os.path.exists(sl2id_file + '.npy'):
    sl2id_np = np.load(sl2id_file + '.npy')
else:
    sl2id_np = np.loadtxt(sl2id_file + '.txt', dtype=np.int64)
    np.save(sl2id_file + '.npy', sl2id_np)
np.random.shuffle(sl2id_np)

# test data from v2
if args.trainv1_testv2:
    version2_notin_version1_file = '../data/version2_notin_version1'
    if os.path.exists(version2_notin_version1_file + '.npy'):
        version2_notin_version1_np = np.load(version2_notin_version1_file + '.npy')
    else:
        version2_notin_version1_np = np.loadtxt(version2_notin_version1_file + '.txt', dtype=np.int64)
        np.save(version2_notin_version1_file + '.npy', version2_notin_version1_np)
    np.random.shuffle(version2_notin_version1_np)

def split_by_genes(sl2id_np, test_size=0.2):
    """Split dataset such that test pairs have only one gene in training set"""
    # Get all unique genes
    all_genes = np.unique(np.concatenate([sl2id_np[:, 0], sl2id_np[:, 1]]))
    
    # First split genes into train and test genes
    n_train_genes = int(len(all_genes) * (1 - test_size))
    train_genes = set(np.random.choice(all_genes, size=n_train_genes, replace=False))
    test_genes = set(all_genes) - train_genes
    
    # Find pairs where exactly one gene is in train_genes
    test_mask = np.array([(pair[0] in train_genes) != (pair[1] in train_genes) for pair in sl2id_np])
    test_pairs = sl2id_np[test_mask]
    
    # Get train pairs (both genes in train_genes)
    train_mask = np.array([(pair[0] in train_genes) and (pair[1] in train_genes) for pair in sl2id_np])
    train_pairs = sl2id_np[train_mask]
    
    # Ensure we have enough test pairs
    min_test_pairs = int(len(sl2id_np) * test_size)
    if len(test_pairs) < min_test_pairs:
        # If not enough, add some pairs where both genes are in test_genes
        additional_test_mask = np.array([(pair[0] in test_genes) and (pair[1] in test_genes) for pair in sl2id_np])
        additional_test_pairs = sl2id_np[additional_test_mask]
        
        if len(additional_test_pairs) > 0:
            needed = min_test_pairs - len(test_pairs)
            if len(additional_test_pairs) > needed:
                additional_test_pairs = additional_test_pairs[np.random.choice(len(additional_test_pairs), needed, replace=False)]
            test_pairs = np.concatenate([test_pairs, additional_test_pairs])
            train_pairs = sl2id_np[np.logical_not(np.isin(sl2id_np, test_pairs).all(axis=1))]
    
    return train_pairs, test_pairs

# Split dataset with gene-based splitting
train_data, test_data = split_by_genes(sl2id_np, test_size=0.2)

if args.trainv1_testv2:
    train_data = sl2id_np
    test_data = version2_notin_version1_np

data = list(data)
data.append(train_data)
data.append(test_data)
data = tuple(data) # n_nodea, n_nodeb, n_entity, n_relation, adj_entity(4), adj_relation(5), train_data(6), test_data(7)

# Initialize metrics lists
train_auc_kkf_list = []
train_f1_kkf_list = []
train_aupr_kkf_list = []
eval_auc_kkf_list = []
eval_f1_kkf_list = []
eval_aupr_kkf_list = []
test_auc_kkf_list = []
test_f1_kkf_list = []
test_aupr_kkf_list = []
loss_kkf_list = []

# Train and evaluate
loss_kf_mean, train_auc_kf_mean, train_f1_kf_mean, train_aupr_kf_mean, \
eval_auc_kf_mean, eval_f1_kf_mean, eval_aupr_kf_mean, \
test_auc_kf_mean, test_f1_kf_mean, test_aupr_kf_mean = train(args, data, param_name + '_1')

# Store metrics
train_auc_kkf_list.append(train_auc_kf_mean)
train_f1_kkf_list.append(train_f1_kf_mean)
train_aupr_kkf_list.append(train_aupr_kf_mean)
eval_auc_kkf_list.append(eval_auc_kf_mean)
eval_f1_kkf_list.append(eval_f1_kf_mean)
eval_aupr_kkf_list.append(eval_aupr_kf_mean)
test_auc_kkf_list.append(test_auc_kf_mean)
test_f1_kkf_list.append(test_f1_kf_mean)
test_aupr_kkf_list.append(test_aupr_kf_mean)
loss_kkf_list.append(loss_kf_mean)

# Calculate mean metrics
train_auc_kkf_mean = np.mean(train_auc_kkf_list)
train_f1_kkf_mean = np.mean(train_f1_kkf_list)
train_aupr_kkf_mean = np.mean(train_aupr_kkf_list)
eval_auc_kkf_mean = np.mean(eval_auc_kkf_list)
eval_f1_kkf_mean = np.mean(eval_f1_kkf_list)
eval_aupr_kkf_mean = np.mean(eval_aupr_kkf_list)
test_auc_kkf_mean = np.mean(test_auc_kkf_list)
test_f1_kkf_mean = np.mean(test_f1_kkf_list)
test_aupr_kkf_mean = np.mean(test_aupr_kkf_list)
loss_kkf_mean = np.mean(loss_kkf_list)

print('The mean of AUC, AUPR and F1 values on the training data are: %.4f, %.4f, %.4f' % (train_auc_kkf_mean, train_aupr_kkf_mean, train_f1_kkf_mean))
print('The mean of AUC, AUPR and F1 values on the validation data are: %.4f, %.4f, %.4f' % (eval_auc_kkf_mean, eval_aupr_kkf_mean, eval_f1_kkf_mean))
print('The mean of AUC, AUPR and F1 values on the test data are: %.4f, %.4f, %.4f' % (test_auc_kkf_mean, test_aupr_kkf_mean, test_f1_kkf_mean))
print('The mean of training loss is: %.4f' % loss_kkf_mean)