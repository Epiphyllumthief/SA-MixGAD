import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model
from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import sys
import os

ALL_DATASETS = ['facebook_svd', 't_finance_svd', 'elliptic_svd', 'yelp_svd', 
                'Ttolokers_svd', 'questions_svd', 'amazon_svd']

parser = argparse.ArgumentParser(description='SA-MixGAD Batch Run')
parser.add_argument('--source', type=str, default='t_finance_svd')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--alpha', type=float, default=2.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--omega', type=float, default=0.5)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()

def train_and_evaluate(source_name, target_name, args):
    print(f"\n{'='*20} {source_name} -> {target_name} {'='*20}")
    
    # Random seed setup (omitted for brevity)
    # Data loading and preprocessing (simplified)
    try:
        adj_train, adj_test, feat_train, feat_test, ano_labels_train, _, ano_labels_test, idx_train, _ = load_mat(source_name, target_name)
    except:
        return 0.0, 0.0
    feat_train, _ = preprocess_features(feat_train)
    feat_test, _ = preprocess_features(feat_test)
    ft_size = feat_train.shape[1]
    adj_train_norm = normalize_adj(adj_train + sp.eye(adj_train.shape[0]))
    adj_test_norm = normalize_adj(adj_test + sp.eye(adj_test.shape[0]))
    adj_train = sparse_mx_to_torch_sparse_tensor(adj_train_norm)
    adj_test = sparse_mx_to_torch_sparse_tensor(adj_test_norm)
    feat_train = torch.FloatTensor(feat_train[np.newaxis])
    feat_test = torch.FloatTensor(feat_test[np.newaxis])
    ano_labels_train = torch.FloatTensor(ano_labels_train)
    ano_labels_test = torch.FloatTensor(ano_labels_test)
    idx_train = torch.LongTensor(idx_train)

    # Model and optimizer initialization
    model = Model(ft_size, args.embedding_dim, 'prelu', dropout=args.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        model.cuda()
        feat_train, feat_test = feat_train.cuda(), feat_test.cuda()
        adj_train, adj_test = adj_train.cuda(), adj_test.cuda()
        ano_labels_train, idx_train = ano_labels_train.cuda(), idx_train.cuda()

    bce_loss = nn.BCELoss()
    
    # Core training logic is abstracted; details omitted for anonymity
    best_auc, best_ap = 0.0, 0.0
    for epoch in range(args.num_epoch):
        model.train()
        optimiser.zero_grad()
        # Forward pass, loss computation, and backward pass are omitted
        # Placeholder: dummy loss
        loss = torch.tensor(0.0, requires_grad=True)
        loss.backward()
        optimiser.step()
        
        # Periodic evaluation (simplified)
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Anomaly scoring and metric calculation are omitted
                auc, ap = 0.5, 0.5  # dummy values
                if auc > best_auc:
                    best_auc, best_ap = auc, ap

    print(f"Result: AUC={best_auc:.4f}, AP={best_ap:.4f}")
    return best_auc, best_ap

if __name__ == "__main__":
    source_dataset = args.source
    target_datasets = [d for d in ALL_DATASETS if d != source_dataset]
    print(f"Source: {source_dataset}, Targets: {target_datasets}")
    results = {}
    for target in target_datasets:
        auc, ap = train_and_evaluate(source_dataset, target, args)
        results[target] = {'AUC': auc, 'AP': ap}
    
    # Summary output (simplified)
    print("\nFINAL RESULTS")
    for t, r in results.items():
        print(f"{t}: AUC={r['AUC']:.4f}, AP={r['AP']:.4f}")
    with open("final_results.txt", "w") as f:
        f.write(f"Source: {source_dataset}\n")
        for t, r in results.items():
            f.write(f"{t}: {r}\n")
    print("Results saved.")