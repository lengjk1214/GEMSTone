import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score
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

def predict_full_data(adata, image_patch, sen_emb, label, label_dict, model_path, seq_len, num_classes, batch_size, device, output_path):
    full_dataset = SCDataset(adata.X, image_patch, sen_emb, label, num_classes, device)
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
        for data_batch, image_batch, sen_batch, _ in full_loader:
            data_batch = data_batch.to(device)
            image_batch = image_batch.to(device)
            sen_batch = sen_batch.to(device)

            hidden = model(data_batch, image_batch, sen_batch, return_encodings=True)
            hidden_cls = hidden.mean(dim=1)  # (batch_size, feature_dim)
            hidden_features.append(hidden_cls.cpu().numpy())

            logits = model(data_batch, image_batch, sen_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            conf = probs.max(dim=1).values.cpu().numpy()

            pred_finals.extend(preds)
            confidences.extend(conf)

    pred_finals = np.array(pred_finals)
    confidences = np.array(confidences)
    hidden_features = np.concatenate(hidden_features, axis=0)  # (N, feature_dim)

    UNASSIGN_THRES = 0.0
    novel_indices = np.where(confidences < UNASSIGN_THRES)[0]

    pred_list = label_dict[pred_finals].tolist()
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

class SCDataset(Dataset):
    def __init__(self, data, image, sen, label, num_classes, device):
        super().__init__()
        self.data = data
        self.image = image
        self.sen = sen
        self.label = label
        self.num_classes = num_classes
        self.device = device

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (self.num_classes - 2)] = self.num_classes - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(self.device)
        return full_seq, self.image[index], self.sen[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../breast_colo/breastCancer2.h5ad")
    parser.add_argument("--image_path", type=str, default="../breast_colo/breastCancer2.tif")
    parser.add_argument("--model_path", type=str, default='../breast2/ckpts_virchow_sen2/finetune_best.pth')
    parser.add_argument("--output_path", type=str, default='../breast2/ckpts_virchow_sen2/finetune_annotated.h5ad')
    parser.add_argument("--image_patch", type=str, default='../breast2/ckpts_virchow_sen2/image_patch')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=201)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    adata = cell2sen(adata, seed=args.seed, save_name="breast1")

    sc.pp.highly_variable_genes(adata, n_top_genes=200)
    adata = adata[:, adata.var['highly_variable']].copy()
    label_dict, encoded_labels = np.unique(np.array(adata.obs["annotation"]), return_inverse=True)
    label = torch.tensor(encoded_labels)

    image_patch = np.array(gene_image(adata, args.image_path))
    sen_emb = adata.obsm["c2s_emb"]

    # 执行预测
    predict_full_data(
        adata, image_patch, sen_emb, label, label_dict,
        args.model_path, args.seq_len, args.num_classes, args.batch_size,
        device, args.output_path
    )
