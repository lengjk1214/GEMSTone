import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from performer_pytorch import PerformerLM
import scanpy as sc
from scipy.sparse import csr_matrix
import pickle as pkl
from utils import *
from sklearn.metrics import adjusted_rand_score

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=3, help="Number of bins.")
parser.add_argument("--gene_num", type=int, default=200, help="Number of genes.")
parser.add_argument("--epoch", type=int, default=100, help="Number of epochs.")
parser.add_argument("--seed", type=int, default=2021, help="Random seed.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--grad_acc", type=int, default=60, help="Gradient accumulation steps.")
parser.add_argument("--valid_every", type=int, default=1, help="Validation frequency.")
parser.add_argument("--pos_embed", type=bool, default=True, help="Using Gene2vec encoding or not.")
parser.add_argument("--data_path", type=str, default="../breast_colo/breastCancer2.h5ad", help="Data path.")
parser.add_argument("--model_path", type=str, default="panglao_pretrain.pth", help="Pretrained model path.")
parser.add_argument("--ckpt_dir", type=str, default="../breast2/ckpts_raw/", help="Checkpoint directory.")
parser.add_argument("--model_name", type=str, default="finetune", help="Finetuned model name.")
args = parser.parse_args()

if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)
    print(f"Directory '{args.ckpt_dir}' created.")
else:
    print(f"Directory '{args.ckpt_dir}' already exists.")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Constants
SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num
PATIENCE = 30
UNASSIGN_THRES = 0.0

# Dataset class
class SCDataset(Dataset):
    def __init__(self, data,label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]  
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        return full_seq, self.label[index] 

    def __len__(self):
        return self.data.shape[0]

# =========================
# Load and preprocess data
# =========================
print("Loading data...")
adata = sc.read_h5ad(args.data_path)

var_names = pd.Series(adata.var_names)
adata = adata[:, ~var_names.duplicated()].copy()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata, base=10)

print("Selecting highly variable genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=200)
adata = adata[:, adata.var['highly_variable']].copy()

# Labels
label_dict, label = np.unique(np.array(adata.obs["annotation"]), return_inverse=True)
label = torch.from_numpy(label)
data = adata.X


print("Splitting dataset into Train / Val / Test ...")

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
trainval_idx, test_idx = next(sss1.split(data, label))
data_trainval = data[trainval_idx]
label_trainval = label[trainval_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1111111, random_state=args.seed)
train_sub_idx, val_sub_idx = next(sss2.split(data_trainval, label_trainval))
train_idx = trainval_idx[train_sub_idx]
val_idx   = trainval_idx[val_sub_idx]

# Final splits
data_train = data[train_idx]
label_train = label[train_idx]

data_val = data[val_idx]
label_val = label[val_idx]

data_test = data[test_idx]
label_test = label[test_idx]


# =========================
# DataLoader
# =========================
train_dataset = SCDataset(data_train, label_train)
val_dataset   = SCDataset(data_val, label_val)
test_dataset  = SCDataset(data_test, label_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# =========================
# Define model
# =========================
print("Building model...")
model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=args.pos_embed,
)

ckpt = torch.load(args.model_path, map_location=device)
pretrained_state_dict = ckpt["model_state_dict"]

current_state_dict = model.state_dict()

# Load only matched parameters
updated_state_dict = {
    k: v for k, v in pretrained_state_dict.items()
    if k in current_state_dict and v.size() == current_state_dict[k].size()
}
current_state_dict.update(updated_state_dict)
model.load_state_dict(current_state_dict)

# Reinitialize unmatched layers
with torch.no_grad():
    if "token_emb.weight" in pretrained_state_dict:
        print(f"Reinitializing token_emb for new sequence length: {SEQ_LEN}")
        model.token_emb = nn.Embedding(SEQ_LEN, model.dim).to(device)

    if "to_out.weight" in pretrained_state_dict:
        print(f"Reinitializing to_out for new class count: {CLASS}")
        model.to_out = nn.Linear(model.dim, CLASS).to(device)

for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True

model.to_out = nn.Linear(model.dim, CLASS)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=args.learning_rate)
loss_fn = nn.CrossEntropyLoss().to(device)

epoch_results = []

print("Start training...")
max_acc = 0.0
trigger_times = 0

for epoch in range(1, args.epoch + 1):
    # ===== Training =====
    model.train()
    running_loss = 0.0
    cum_acc = 0.0

    for data_batch, labels in train_loader:
        data_batch = data_batch.long().to(device)
        labels = labels.to(device)

        logits = model(data_batch)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        cum_acc += (logits.argmax(dim=-1) == labels).float().mean().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * cum_acc / len(train_loader)
    print(f"Epoch {epoch}: Training Loss: {epoch_loss:.6f}, Accuracy: {epoch_acc:.2f}%")

    epoch_results.append({
        "epoch": epoch,
        "train_loss": epoch_loss,
        "train_acc": epoch_acc
    })

    # ===== Validation =====
    if epoch % args.valid_every == 0:
        model.eval()
        running_loss = 0.0
        predictions = []
        truths = []

        with torch.no_grad():
            for data_batch, labels in val_loader:
                data_batch = data_batch.to(device)
             
                labels = labels.to(device)

                logits = model(data_batch)
                loss = loss_fn(logits, labels)
                running_loss += loss.item()

                preds = logits.argmax(dim=-1)
                predictions.extend(preds.cpu().numpy())
                truths.extend(labels.cpu().numpy())

        val_loss = running_loss / len(val_loader)
        cur_acc = accuracy_score(truths, predictions)
        cur_f1 = f1_score(truths, predictions, average="macro")
        cur_ari = adjusted_rand_score(truths, predictions)

        print(f"Epoch {epoch}: Validation Loss: {val_loss:.6f}, Accuracy: {cur_acc:.4f}, F1 Score: {cur_f1:.4f}, ARI: {cur_ari:.4f}")

        epoch_results[-1].update({
            "val_loss": val_loss,
            "val_acc": cur_acc,
            "val_f1": cur_f1,
            "val_ari": cur_ari
        })

        # ===== Save best model by Validation Accuracy =====
        if cur_acc > max_acc:
            max_acc = cur_acc
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{args.model_name}_best.pth"))
            print(f"Best model saved at epoch {epoch}.")
        else:
            trigger_times += 1
            if trigger_times > PATIENCE:
                print("Early stopping.")
                break

# Save training log
results_df = pd.DataFrame(epoch_results)
results_csv_path = os.path.join(args.ckpt_dir, f"{args.model_name}_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Saved training log to: {results_csv_path}")

# =========================
# Final TEST evaluation
# =========================
print("\nEvaluating on independent TEST set...")

best_model_path = os.path.join(args.ckpt_dir, f"{args.model_name}_best.pth")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

test_predictions = []
test_truths = []

with torch.no_grad():
    for data_batch, labels in test_loader:
        data_batch = data_batch.to(device)
        labels = labels.to(device)

        logits = model(data_batch)
        preds = logits.argmax(dim=-1)

        test_predictions.extend(preds.cpu().numpy())
        test_truths.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_truths, test_predictions)
test_f1 = f1_score(test_truths, test_predictions, average="macro")
test_ari = adjusted_rand_score(test_truths, test_predictions)

print("\n" + "=" * 60)
print("FINAL TEST SET PERFORMANCE")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test ARI: {test_ari:.4f}")
print("=" * 60 + "\n")

test_metrics = {
    "test_acc": test_acc,
    "test_f1": test_f1,
    "test_ari": test_ari
}
with open(os.path.join(args.ckpt_dir, f"{args.model_name}_test_metrics.pkl"), "wb") as f:
    pkl.dump(test_metrics, f)

print("Training and evaluation completed successfully.")