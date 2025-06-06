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

param_name = 'final'
args = parser.parse_args()
data = load_data(args)
kf = ShuffleSplit(n_splits=10, test_size=0.2, random_state=43)

# Reading data files
print('reading sl2id file again for splitting dataset...')
sl2id_file = '../data/sl2id'
if os.path.exists(sl2id_file + '.npy'):
    sl2id_np = np.load(sl2id_file + '.npy')
else:
    sl2id_np = np.loadtxt(sl2id_file + '.txt', dtype=np.int64)
    np.save(sl2id_file + '.npy', sl2id_np)
np.random.shuffle(sl2id_np)

# Test data from v2
if args.trainv1_testv2:
    version2_notin_version1_file = '../data/version2_notin_version1'
    if os.path.exists(version2_notin_version1_file + '.npy'):
        version2_notin_version1_np = np.load(version2_notin_version1_file + '.npy')
    else:
        version2_notin_version1_np = np.loadtxt(version2_notin_version1_file + '.txt', dtype=np.int64)
        np.save(version2_notin_version1_file + '.npy', version2_notin_version1_np)
    np.random.shuffle(version2_notin_version1_np)

k = 1
metrics = {
    'train': {'auc': [], 'f1': [], 'aupr': []},
    'val': {'auc': [], 'f1': [], 'aupr': []},
    'test': {'auc': [], 'f1': [], 'aupr': []},
    'loss': []
}

for train_data, test_data in kf.split(sl2id_np):
    tf.reset_default_graph()

    if not args.trainv1_testv2:
        train_data = sl2id_np[train_data]
        test_data = sl2id_np[test_data]

    if args.trainv1_testv2:
        train_data = sl2id_np
        test_data = version2_notin_version1_np

    data = list(data)
    data.append(train_data)
    data.append(test_data)
    data = tuple(data)

    # Modified train function should return all metrics
    result = train(args, data, param_name + '_' + str(k))
    
    # Unpack results
    (loss_kf_mean, 
     train_auc_kf_mean, train_f1_kf_mean, train_aupr_kf_mean, 
     val_auc_kf_mean, val_f1_kf_mean, val_aupr_kf_mean, 
     test_auc_kf_mean, test_f1_kf_mean, test_aupr_kf_mean) = result

    # Store metrics
    metrics['train']['auc'].append(train_auc_kf_mean)
    metrics['train']['f1'].append(train_f1_kf_mean)
    metrics['train']['aupr'].append(train_aupr_kf_mean)
    
    metrics['val']['auc'].append(val_auc_kf_mean)
    metrics['val']['f1'].append(val_f1_kf_mean)
    metrics['val']['aupr'].append(val_aupr_kf_mean)
    
    metrics['test']['auc'].append(test_auc_kf_mean)
    metrics['test']['f1'].append(test_f1_kf_mean)
    metrics['test']['aupr'].append(test_aupr_kf_mean)
    
    metrics['loss'].append(loss_kf_mean)
    
    k += 1
    if k > 1:  # For testing, run only one fold
        break

# Calculate mean metrics
def calculate_mean(metric_list):
    return np.mean(metric_list) if metric_list else 0.0

print('\nFinal Results:')
print('-' * 50)
print('Training:')
print(f"AUC: {calculate_mean(metrics['train']['auc']):.4f} | "
      f"F1: {calculate_mean(metrics['train']['f1']):.4f} | "
      f"AUPR: {calculate_mean(metrics['train']['aupr']):.4f}")

print('\nValidation:')
print(f"AUC: {calculate_mean(metrics['val']['auc']):.4f} | "
      f"F1: {calculate_mean(metrics['val']['f1']):.4f} | "
      f"AUPR: {calculate_mean(metrics['val']['aupr']):.4f}")

print('\nTesting:')
print(f"AUC: {calculate_mean(metrics['test']['auc']):.4f} | "
      f"F1: {calculate_mean(metrics['test']['f1']):.4f} | "
      f"AUPR: {calculate_mean(metrics['test']['aupr']):.4f}")

print('\nTraining Loss:')
print(f"Mean: {calculate_mean(metrics['loss']):.4f}")
print('-' * 50)