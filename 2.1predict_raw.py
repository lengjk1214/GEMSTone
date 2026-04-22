import torch
import numpy as np
import scanpy as sc
import pandas as pd
import pickle as pkl
import argparse
from performer_pytorch import PerformerLM
from torch import nn
import timm
from torchvision import transforms
from timm.layers import SwiGLUPacked
import cv2
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader, Dataset
import random
import cell2sentence as cs
from cell2sentence.tasks import embed_cells
from scipy.optimize import linear_sum_assignment 


def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
    preds_k = len(np.unique(flat_preds)) if preds_k is None else preds_k
    target_k = len(np.unique(flat_target)) if target_k is None else target_k

    num_correct = np.zeros((preds_k, target_k), dtype=np.int32)
    for c1 in range(preds_k):
        for c2 in range(target_k):
            num_correct[c1, c2] = int(((flat_preds == c1) & (flat_target == c2)).sum())

    total_samples = flat_preds.shape[0]
    cost_matrix = total_samples - num_correct

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    match = list(zip(row_ind, col_ind)) 

    matched_targets = set(col for _, col in match)
    all_targets = set(range(target_k))
    unmatched_targets = all_targets - matched_targets

    for gt_c in unmatched_targets:
        match.append((-1, gt_c))  

    return match

def predict_full_data(adata, label, label_dict, model_path, seq_len, num_classes, batch_size, device, output_path):
    full_dataset = SCDataset(adata.X, label, num_classes)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    model = PerformerLM(
        num_tokens=num_classes,
        dim=200,
        depth=6,
        max_seq_len=seq_len,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=True,
    )
    model.token_emb = nn.Embedding(seq_len, model.dim).to(device)
    model.to_out = nn.Linear(model.dim, num_classes)
    model = model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    pred_finals = []
    confidences = []
    hidden_features = []

    with torch.no_grad():
        for data_batch, _ in full_loader:
            data_batch = data_batch.to(device)

            hidden = model(data_batch, return_encodings=True)
            hidden_cls = hidden.mean(dim=1)  # (batch_size, feature_dim)
            hidden_features.append(hidden_cls.cpu().numpy())

            logits = model(data_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            conf = probs.max(dim=1).values.cpu().numpy()

            pred_finals.extend(preds)
            confidences.extend(conf)

    pred_finals = np.array(pred_finals)
    confidences = np.array(confidences)
    hidden_features = np.concatenate(hidden_features, axis=0)  # (N, feature_dim)

    flat_target = encoded_labels
    preds_k = len(label_dict)     
    target_k = len(label_dict)    

    match_results = _hungarian_match(pred_finals, flat_target, preds_k, target_k)

    match_dict = {pred_idx: gt_idx for pred_idx, gt_idx in match_results}

    pred_finals_matched = np.array([match_dict[pred_idx] for pred_idx in pred_finals])

    UNASSIGN_THRES = 0.0
    novel_indices = np.where(confidences < UNASSIGN_THRES)[0]

    pred_list = label_dict[pred_finals_matched].tolist()
    for idx in novel_indices:
        pred_list[idx] = "Unassigned"

    adata.obs["predicted_cell_type"] = pred_list
    adata.obsm["pred"] = hidden_features

    assigned_mask = adata.obs["predicted_cell_type"] != "Unassigned"
    if np.any(assigned_mask):
        ari = adjusted_rand_score(
            adata.obs["annotation"][assigned_mask],
            adata.obs["predicted_cell_type"][assigned_mask]
        )
        print(f"Final prediction ARI (excluding 'Unassigned'): {ari:.4f}")
    else:
        print("All predictions are 'Unassigned', ARI not computed.")

    adata.write_h5ad(output_path)
    print(f"Annotated adata saved to: {output_path}")


class SCDataset(Dataset):
    def __init__(self, data,label, num_classes):
        super().__init__()
        self.data = data
        self.label = label
        self.num_classes = num_classes

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]  
        full_seq[full_seq > (self.num_classes - 2)] = self.num_classes - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        return full_seq, self.label[index] 
    

    def __len__(self):
        return self.data.shape[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../breast_colo/colorectal-cancer.h5ad")
    parser.add_argument("--model_path", type=str, default='../colo/ckpts_raw/finetune_best.pth')
    parser.add_argument("--output_path", type=str, default='../colo/ckpts_raw/finetune_annotated.h5ad')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=201)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.data_path)
    var_names = pd.Series(adata.var_names)
    adata = adata[:, ~var_names.duplicated()] 

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=200)
    adata = adata[:, adata.var['highly_variable']].copy()
    label_dict, encoded_labels = np.unique(np.array(adata.obs["annotation"]), return_inverse=True)
    label = torch.tensor(encoded_labels)

    predict_full_data(
        adata, label, label_dict,
        args.model_path, args.seq_len, args.num_classes, args.batch_size,
        device, args.output_path)