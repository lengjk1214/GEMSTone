
import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from performer_pytorch import PerformerLM
import scanpy as sc
from scipy.sparse import csr_matrix
import pickle as pkl
from utils import *
import timm
from torchvision import transforms
import timm
from timm.layers import SwiGLUPacked
import cv2
from sklearn.metrics import adjusted_rand_score
import cell2sentence as cs
from cell2sentence.tasks import embed_cells


parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=4, help="Number of class.")
parser.add_argument("--gene_num", type=int, default=200, help="Number of genes.")
parser.add_argument("--epoch", type=int, default=100, help="Number of epochs.")
parser.add_argument("--seed", type=int, default=2021, help="Random seed.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--grad_acc", type=int, default=60, help="Gradient accumulation steps.")
parser.add_argument("--valid_every", type=int, default=1, help="Validation frequency.")
parser.add_argument("--pos_embed", type=bool, default=True, help="Using Gene2vec encoding or not.")
parser.add_argument("--data_path", type=str, default="../breast_colo/breastCancer1.h5ad", help="Data path.")
parser.add_argument("--image_path", type=str, default="../breast_colo/breastCancer1.tif", help="Image path.")
parser.add_argument("--model_path", type=str, default="panglao_pretrain.pth", help="Pretrained model path.")
parser.add_argument("--ckpt_dir", type=str, default="../breast1/ckpts_spatial2_sen2/", help="Checkpoint directory.")
parser.add_argument("--model_name", type=str, default="finetune", help="Finetuned model name.")
args = parser.parse_args()

if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)
    print(f"Directory '{args.ckpt_dir}' created.")
else:
    print(f"Directory '{args.ckpt_dir}' already exists.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num
PATIENCE = 50
UNASSIGN_THRES = 0.0

class SCDataset(Dataset):
    def __init__(self, data,image,sen, label):
        super().__init__()
        self.data = data
        self.image = image
        self.sen = sen
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0] - 1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        return full_seq, self.image[rand_start],self.sen[rand_start], self.label[rand_start]

    def __len__(self):
        return self.data.shape[0]

import torch
import cv2
import numpy as np
from torchvision import transforms
import timm
from typing import List

def gene_image(adata, image_path, alpha = 0.05):
    checkpoint_path = "../vrichow.bin"
    img_size = 224 
    reg_tokens = 4  
    
    model = timm.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        checkpoint_path=checkpoint_path,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        img_size=img_size,
        init_values=1e-5,
        num_classes=0,
        reg_tokens=reg_tokens,
        mlp_ratio=5.3375,
        global_pool="",
        dynamic_img_size=True,
    )
    model.eval()
    model = model.to("cuda")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to("cuda")  

    patch_features = []
    w, h = 60, 60
    for i, coor in enumerate(adata.obsm['spatial_scale']):
        # flaten
        # x, y = coor
        # img_p = image[:, int(y-h):int(y+h), int(x-w): int(x+w)]
        # feature = img_p.flatten()
        # patch_features.append(feature)

        x, y = coor
        img_p = image_tensor[0, :, int(y-h):int(y+h), int(x-w):int(x+w)]
        resize = transforms.Resize((224, 224))
        img_resized = resize(img_p.unsqueeze(0))
        
        with torch.inference_mode():
            output = model(img_resized.to(torch.float32))
        
        class_token = output[:, 0] 
        patch_tokens = output[:, 5:] 
        local_feat = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
        patch_features.append(local_feat.detach().cpu())

    local_features = torch.cat(patch_features, dim=0).numpy() 

    resize_global = transforms.Resize((224, 224))
    image_global = resize_global(image_tensor[0]).unsqueeze(0).to("cuda")

    with torch.inference_mode():
        output_global = model(image_global.to(torch.float32))

    class_token_global = output_global[:, 0]
    patch_tokens_global = output_global[:, 5:]
    global_feat = torch.cat([class_token_global, patch_tokens_global.mean(dim=1)], dim=-1)
    global_feat_np = global_feat.detach().cpu().numpy().flatten()

    combined_features = local_features + (alpha * global_feat_np)

    return [f for f in combined_features]


def get_top_genes(adata, n_top=30):
    X = adata.X
    if isinstance(X, np.ndarray):
        X = csr_matrix(X)
    top_genes_dict = {}
    for index in range(X.shape[0]):
        row_data = X[index].toarray().ravel()
        
        top_indices = np.argsort(-row_data)[:n_top]
        
        gene_names = adata.var.index[top_indices].tolist()
        top_genes_dict[index] = gene_names
    return top_genes_dict

def cell_to_sentence(adata, n_top=30, obs_features=None):
    top_genes_dict = get_top_genes(adata, n_top) 
    formatted_dict = {}
    
    for cell_id, genes in top_genes_dict.items():
        if len(genes) > 1:
            gene_sentence = "Top genes are {}, and {}.".format(", ".join(genes[:-1]), genes[-1])
        else:
            gene_sentence = "Top gene is {}.".format(genes[0])
        
        if obs_features:
            for feature in obs_features:
                if feature in adata.obs:
                    feature_value = str(adata.obs[feature].iloc[cell_id])
                    gene_sentence += " {} of this cell is {}.".format(feature, feature_value)
        
        formatted_dict[cell_id] = gene_sentence
    return formatted_dict


def lm_cell_embed(adata_, top_k=30, model_name="all-MiniLM-L6-v2",gene_list=None, obs_features=None,
                 return_sentence=True):
    embedding_function = SentenceTransformer('../all-MiniLM-L12-v2')
    if gene_list is not None:
        adata = adata_[:,gene_list].copy()
    else:
        adata = adata_.copy()
    top_genes_sentences = cell_to_sentence(adata, top_k, obs_features)
    sentences = list(top_genes_sentences.values())
    db = embedding_function.encode(sentences, convert_to_tensor=False)  # 返回 NumPy 数组
    emb_res = np.asarray(db)
    
    adata_emb = sc.AnnData(emb_res)
    adata_emb.var= pd.DataFrame(range(emb_res.shape[1]))
    adata_emb.obs = adata.obs
    sc.tl.pca(adata_emb, svd_solver='arpack')
    sc.pp.neighbors(adata_emb, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata_emb)
    
    if return_sentence:
        adata_emb.obs['cell_sentence']= sentences
    return adata_emb

def cell2sen(adata, seed=42, top_k_genes=100, save_dir="../cell2sentence/data", 
             save_name="breast1", model_path="../C2S-Pythia-410m-cell-type-prediction"):
    adata_obs_cols_to_keep = ["tissue", "cell_type", "organism"]
    
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata, 
        random_state=seed, 
        sentence_delimiter=' ',
        label_col_names=adata_obs_cols_to_keep
    )
    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds, 
        vocabulary=vocabulary,
        save_dir=save_dir,
        save_name=save_name,
        dataset_backend="arrow"
    )
    csmodel = cs.CSModel(
        model_name_or_path=model_path,
        save_dir=save_dir,
        save_name=save_name
    )
    embedded_cells = embed_cells(
        csdata=csdata,
        csmodel=csmodel,
        n_genes=top_k_genes,
    )
    adata.obsm["c2s_emb"] = embedded_cells
    return adata

adata = sc.read_h5ad(args.data_path)
var_names = pd.Series(adata.var_names)
adata = adata[:, ~var_names.duplicated()]
adata.obs['cell_type'] = adata.obs['annotation']
adata.obs['tissue'] = adata.obs['in_tissue']
adata.obs['organism'] = 'Homo sapiens'


sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata, base=10)  
# adata_emb=  lm_cell_embed(adata, top_k=20, model_name="all-MiniLM-L12-v2",
#                           gene_list=None, obs_features= None,
#                          return_sentence=True)
# sen_emb = adata_emb.X
adata = cell2sen(adata, seed=args.seed, save_dir=args.ckpt_dir, save_name="cell2sen")
sen_emb = adata.obsm["c2s_emb"]

sc.pp.highly_variable_genes(adata, n_top_genes=200)
adata = adata[:, adata.var['highly_variable']].copy()
label_dict, label = np.unique(np.array(adata.obs["annotation"]), return_inverse=True)
label = torch.from_numpy(label)
data = adata.X
image_patch = np.array(gene_image(adata, args.image_path))

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
image_train = image_patch[train_idx]
sen_train = sen_emb[train_idx]
label_train = label[train_idx]

data_val = data[val_idx]
image_val = image_patch[val_idx]
sen_val = sen_emb[val_idx]
label_val = label[val_idx]

data_test = data[test_idx]
image_test = image_patch[test_idx]
sen_test = sen_emb[test_idx]
label_test = label[test_idx]

# =========================
# DataLoader
# =========================
train_dataset = SCDataset(data_train, image_train, sen_train, label_train)
val_dataset   = SCDataset(data_val, image_val, sen_val, label_val)
test_dataset  = SCDataset(data_test, image_test, sen_test, label_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print("Building model...")
model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=args.pos_embed,
    image_dim=image_patch.shape[1],
    sen_dim=sen_emb.shape[1]
)

ckpt = torch.load(args.model_path, map_location=device)
pretrained_state_dict = ckpt["model_state_dict"]

current_state_dict = model.state_dict()

updated_state_dict = {
    k: v for k, v in pretrained_state_dict.items()
    if k in current_state_dict and v.size() == current_state_dict[k].size()
}
current_state_dict.update(updated_state_dict)
model.load_state_dict(current_state_dict)

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

    for data_batch, image_batch, sen_batch, labels in train_loader:
        data_batch = data_batch.long().to(device)
        labels = labels.to(device)
        image_batch = image_batch.to(device)
        sen_batch = sen_batch.to(device)

        logits = model(data_batch, image_batch, sen_batch)
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
            for data_batch, image_batch, sen_batch, labels in val_loader:
                data_batch = data_batch.to(device)
                image_batch = image_batch.to(device)
                sen_batch = sen_batch.to(device)
                labels = labels.to(device)

                logits = model(data_batch, image_batch, sen_batch)
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

print("\nEvaluating on independent TEST set...")

best_model_path = os.path.join(args.ckpt_dir, f"{args.model_name}_best.pth")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

test_predictions = []
test_truths = []

with torch.no_grad():
    for data_batch, image_batch, sen_batch, labels in test_loader:
        data_batch = data_batch.to(device)
        image_batch = image_batch.to(device)
        sen_batch = sen_batch.to(device)
        labels = labels.to(device)

        logits = model(data_batch, image_batch, sen_batch)
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

# 保存最终 test 指标
test_metrics = {
    "test_acc": test_acc,
    "test_f1": test_f1,
    "test_ari": test_ari
}
with open(os.path.join(args.ckpt_dir, f"{args.model_name}_test_metrics.pkl"), "wb") as f:
    pkl.dump(test_metrics, f)

print("Training and evaluation completed successfully.")