import os
import sys
import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

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
parser.add_argument('--test_cell_line', type=str, default='RPE1', help='cell line to use as test set')
args = parser.parse_args()

param_name = 'final'
data = load_data(args)

# Load gene pairs data with cell line information
print('Reading sl2id file with cell line information...')
sl2id_file = '../data/SLKB_rawSL.csv'  # Assuming this file contains cell line information
if os.path.exists(sl2id_file):
    sl_data = pd.read_csv(sl2id_file)
else:
    raise FileNotFoundError(f"Could not find file: {sl2id_file}")

# Filter data for the specified cell line
test_data = sl_data[sl_data['cell_line_origin'] == args.test_cell_line]
train_val_data = sl_data[sl_data['cell_line_origin'] != args.test_cell_line]

# Split train_val_data into training and validation sets
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

# Convert to numpy arrays (assuming we need gene pairs in numpy format)
train_data_np = train_data[['gene_1_id', 'gene_2_id', 'type']].to_numpy()
val_data_np = val_data[['gene_1_id', 'gene_2_id', 'type']].to_numpy()
test_data_np = test_data[['gene_1_id', 'gene_2_id', 'type']].to_numpy()

# Prepare data for training
data = list(data)
data.append(train_data_np)
data.append(test_data_np)
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

# Print split information
print(f"\nDataset Split Information:")
print(f"Test Cell Line: {args.test_cell_line}")
print(f"Training samples: {len(train_data_np)}")
print(f"Validation samples: {len(val_data_np)}")
print(f"Test samples: {len(test_data_np)}")

# Calculate mean metrics
print('\nFinal Results:')
print('Training -   AUC: %.4f, AUPR: %.4f, F1: %.4f' % (
    np.mean(metrics['train_auc']), np.mean(metrics['train_aupr']), np.mean(metrics['train_f1'])))
print('Validation - AUC: %.4f, AUPR: %.4f, F1: %.4f' % (
    np.mean(metrics['eval_auc']), np.mean(metrics['eval_aupr']), np.mean(metrics['eval_f1'])))
print('Test -       AUC: %.4f, AUPR: %.4f, F1: %.4f' % (
    np.mean(metrics['test_auc']), np.mean(metrics['test_aupr']), np.mean(metrics['test_f1'])))
print('Training loss: %.4f' % np.mean(metrics['loss']))