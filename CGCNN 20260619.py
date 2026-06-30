"""
MagGNoME v6.9: Nested Folders Edition (Smoke Test Configured)
=============================================================
- Recursively scans './extracted_batches' and all sub-folders for .cif files.
- EPOCHS and MC_PASSES reduced to 1 for rapid pipeline verification.
- Replaced C++ radius_graph with pure PyTorch matrix math.
- Fixed PicklingError by unpacking emmet-core MPDataDocs into standard dicts.
- Fixed global_mean_pool tensor collapse by explicitly passing size=n_graphs.
- Swaps all JSON outputs for clean, Jupyter-accessible CSV files.
"""

import os
import sys
import json
import pickle
import glob
import math
import contextlib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MessagePassing

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: savefig works, plt.show() won't block
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

import warnings
warnings.filterwarnings("ignore")

# ── CONFIGURATION ────────────────────────────────────────────────────────────
MP_API_KEY          = os.environ.get("MP_API_KEY", "FJ9RiSfrTX30wN1yppawNgokMc8nuHxo")
GNOME_MASTER_DIR    = "./extracted_batches"  # Hardcoded to your specific folder
GRAPH_CACHE_DIR     = "./graph_cache_mp_cgcnn"
GNOME_CACHE_DIR     = "./graph_cache_gnome_cgcnn"

CUTOFF_RADIUS       = 4.5  
MAX_UNIT_CELL_ATOMS = 80

CGCNN_HIDDEN        = 128
CGCNN_BLOCKS        = 4
SG_EMB_DIM          = 16
SUBLAT_DIM          = 64
ATTN_HEADS          = 4

AUX_WEIGHT          = 0.30

# Jupyter Execution Settings (SMOKE TEST CONFIGURATION)
BATCH_SIZE          = 16
NUM_WORKERS         = 0     
EPOCHS              = 50    # Real training run
LR                  = 1e-3
PATIENCE            = 10
MC_PASSES           = 30    # Real MC-dropout uncertainty estimate

MGOe_TO_Jm3         = 7957.75
MAGNITO_THRESHOLD   = 60 * MGOe_TO_Jm3 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RARE_EARTH_Z = {21, 39} | set(range(57, 72))
TM_Z         = {22, 23, 24, 25, 26, 27, 28}
MU0          = 4 * np.pi * 1e-7


# ── PHYSICS & MATH UTILS ─────────────────────────────────────────────────────
def bhmax_from_moment(m: float) -> float:
    msat = abs(m) * 9.2740100783e6   # μB/Å³ → A/m (μ_B 9.274e-24 A·m² / Å³ 1e-30 m³); was 1e3 (wrong, off ~9274×)            
    bsat = MU0 * msat              
    return bsat ** 2 / (4 * MU0)   

@contextlib.contextmanager
def _silence_c_stderr():
    """spglib writes its 'failed' notes straight to fd 2, bypassing Python warnings."""
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), fd)
        try:
            yield
        finally:
            os.dup2(saved, fd)
            os.close(saved)

def get_spacegroup_number(struct: Structure) -> int:
    try:
        with _silence_c_stderr():
            return SpacegroupAnalyzer(struct, symprec=0.2, angle_tolerance=5.0).get_space_group_number()
    except Exception:
        return 1

def robust_radius_graph(pos, r, batch):
    dist_mat = torch.cdist(pos, pos, p=2.0)
    batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
    valid_edges = (dist_mat <= r) & (dist_mat > 1e-5) & batch_mask
    edge_index = valid_edges.nonzero(as_tuple=False).t().contiguous()
    return edge_index


# ── CGCNN ARCHITECTURE ───────────────────────────────────────────────────────
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=64):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class CGCNNConv(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='add')
        self.fc_full = nn.Linear(2 * node_dim + edge_dim, 2 * node_dim)
        self.bn1 = nn.BatchNorm1d(2 * node_dim)
        self.bn2 = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        z = self.bn1(self.fc_full(z))
        core, filter_gate = z.chunk(2, dim=-1)
        return torch.sigmoid(filter_gate) * F.softplus(core)

    def update(self, aggr_out, x):
        return self.bn2(x + aggr_out)

class CGCNNAtomwise(nn.Module):
    def __init__(self, hidden_channels=CGCNN_HIDDEN, num_blocks=CGCNN_BLOCKS, cutoff=CUTOFF_RADIUS):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, 64)
        self.convs = nn.ModuleList([CGCNNConv(hidden_channels, 64) for _ in range(num_blocks)])

    def forward(self, z, pos, batch):
        x = self.embedding(z)
        edge_index = robust_radius_graph(pos, r=self.cutoff, batch=batch)
        edge_weight = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], p=2, dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return x

class SublatticeAttentionPool(nn.Module):
    def __init__(self, hidden_dim=CGCNN_HIDDEN, sublat_dim=SUBLAT_DIM, sg_emb_dim=SG_EMB_DIM, n_heads=ATTN_HEADS):
        super().__init__()
        self.re_proj  = nn.Linear(hidden_dim, sublat_dim)
        self.tm_proj  = nn.Linear(hidden_dim, sublat_dim)
        self.all_proj = nn.Linear(hidden_dim, sublat_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=sublat_dim, num_heads=n_heads, dropout=0.1, batch_first=True)
        self.attn_norm  = nn.LayerNorm(sublat_dim)
        self.global_proj = nn.Linear(sg_emb_dim, sublat_dim)   # was sg_emb_dim+1; dropped global_feats (magnetization = label leakage)
        self.out_proj    = nn.Linear(sublat_dim * 2, sublat_dim)

    def forward(self, x, batch, is_RE, is_TM, sg_emb, n_graphs):
        device = x.device
        
        re_repr = global_mean_pool(self.re_proj(x[is_RE]), batch[is_RE], size=n_graphs) if is_RE.any() else torch.zeros(n_graphs, SUBLAT_DIM, device=device)
        non_RE = ~is_RE
        
        if is_TM.any(): 
            tm_repr = global_mean_pool(self.tm_proj(x[is_TM]), batch[is_TM], size=n_graphs)
        elif non_RE.any(): 
            tm_repr = global_mean_pool(self.all_proj(x[non_RE]), batch[non_RE], size=n_graphs)
        else: 
            tm_repr = re_repr.detach().clone()

        re_q, tm_k = re_repr.unsqueeze(1), tm_repr.unsqueeze(1)
        attended, _ = self.cross_attn(re_q, tm_k, tm_k)
        re_attended = self.attn_norm(re_repr + attended.squeeze(1))

        combined = self.out_proj(torch.cat([re_attended, self.global_proj(sg_emb)], dim=-1))
        re_dominance = ((1.0 - F.cosine_similarity(re_repr, tm_repr, dim=-1)) / 2.0).clamp(0.0, 1.0)       
        return combined, re_dominance

class MagGNoME(nn.Module):
    def __init__(self):
        super().__init__()
        self.cgcnn = CGCNNAtomwise()
        self.sg_embedding = nn.Embedding(232, SG_EMB_DIM)
        self.sublattice_pool = SublatticeAttentionPool()
        self.dropout = nn.Dropout(p=0.15)
        self.bhmax_head = nn.Sequential(nn.Linear(SUBLAT_DIM, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 1), nn.Softplus())
        self.site_moment_head = nn.Sequential(nn.Linear(CGCNN_HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, data, mc_dropout=False, return_dominance=False):
        atom_repr = self.dropout(self.cgcnn(data.z, data.pos, data.batch)) if mc_dropout else self.cgcnn(data.z, data.pos, data.batch)
        sg_emb = self.sg_embedding(data.sg.squeeze(-1))
        n_graphs = data.batch.max().item() + 1
        if sg_emb.dim() == 1: sg_emb = sg_emb.unsqueeze(0)

        crystal_repr, re_dominance = self.sublattice_pool(atom_repr, data.batch, data.is_RE, data.is_TM, sg_emb, n_graphs)

        bhmax_pred, site_m_pred = self.bhmax_head(crystal_repr), self.site_moment_head(atom_repr)     
        return (bhmax_pred, site_m_pred, re_dominance) if return_dominance else (bhmax_pred, site_m_pred)

class CombinedMagLoss(nn.Module):
    def forward(self, u_pred, u_true, site_pred, site_true, has_site_labels, batch_idx):
        base = F.huber_loss(u_pred, u_true, delta=1.0, reduction="none")
        primary = ((1.0 + 2.0 * (u_true >= MAGNITO_THRESHOLD).float() * (u_pred < u_true).float()) * base).mean()
        
        atom_has_label = has_site_labels.squeeze(-1)[batch_idx] 
        if not atom_has_label.any() or AUX_WEIGHT == 0.0: return primary

        aux = F.huber_loss(site_pred.squeeze(-1)[atom_has_label], site_true[atom_has_label], delta=1.0, reduction="mean")
        return primary + AUX_WEIGHT * aux


# ── CACHING & DATALOADER ─────────────────────────────────────────────────────
def _build_and_cache_one(struct, total_moment, site_moments, mat_id, cache_dir):
    cache_path = os.path.join(cache_dir, f"{mat_id}.pt")
    if os.path.exists(cache_path): return True
    z_numbers = [site.specie.number for site in struct]
    if not any(z in RARE_EARTH_Z for z in z_numbers): return False

    n_atoms = len(struct)
    z   = torch.tensor(z_numbers, dtype=torch.long)
    pos = torch.tensor(np.array([site.coords for site in struct], dtype=np.float32), dtype=torch.float)

    if site_moments is not None and len(site_moments) == n_atoms:
        raw = [float(m) for m in site_moments]
        m_sites, has_site_labels = (np.zeros(n_atoms, dtype=np.float32), False) if any(math.isnan(v) for v in raw) else (np.array(raw, dtype=np.float32), True)
    else:
        m_sites, has_site_labels = np.zeros(n_atoms, dtype=np.float32), False

    max_abs = np.max(np.abs(m_sites)) + 1e-8
    m_scale = torch.tensor([max_abs], dtype=torch.float)

    moment_known = total_moment is not None
    y = torch.tensor([[bhmax_from_moment(float(total_moment))]], dtype=torch.float) if moment_known and not np.isnan(float(total_moment)) else None

    data = Data(z=z, pos=pos, sg=torch.tensor([get_spacegroup_number(struct)], dtype=torch.long), 
                global_feats=torch.tensor([math.log1p(abs(float(total_moment))) if moment_known else 0.0], dtype=torch.float), 
                global_feats_known=torch.tensor([moment_known], dtype=torch.bool), y=y, 
                site_m=torch.tensor(m_sites, dtype=torch.float), site_m_norm=torch.tensor(m_sites / max_abs, dtype=torch.float), 
                m_scale=m_scale, is_RE=torch.tensor([z_num in RARE_EARTH_Z for z_num in z_numbers], dtype=torch.bool), 
                is_TM=torch.tensor([z_num in TM_Z for z_num in z_numbers], dtype=torch.bool),
                has_site_labels=torch.tensor([has_site_labels], dtype=torch.bool), mat_id=mat_id)
    torch.save(data, cache_path)
    return True

def _write_index(cache_dir: str) -> None:
    index = {}
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.pt"))):
        try:
            d = torch.load(f, weights_only=False)
            index[os.path.basename(f)] = {"has_label": d.y is not None, "umag": float(d.y.item()) if d.y is not None else None}
        except Exception: pass
    with open(os.path.join(cache_dir, "index.json"), "w") as fp: json.dump(index, fp)

class LazyGraphDataset(PyGDataset):
    def __init__(self, cache_dir, labelled_only=True):
        super().__init__()
        idx_path = os.path.join(cache_dir, "index.json")
        if not os.path.exists(idx_path): _write_index(cache_dir)
        with open(idx_path) as fp: index = json.load(fp)
        self.files = sorted([os.path.join(cache_dir, fname) for fname, meta in index.items() if not labelled_only or meta["has_label"]])
    def len(self): return len(self.files)
    def get(self, idx): return torch.load(self.files[idx], weights_only=False)


# ── EVALUATION & PLOTTING ────────────────────────────────────────────────────
def plot_parity(y_true, y_pred, out_path="parity_validation.png"):
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', color='seagreen')
    min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel("True (BH)max (MGOe)")
    plt.ylabel("Predicted (BH)max (MGOe)")
    plt.title("CGCNN Parity Plot — Validation Set (SMOKE TEST)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    # plt.show()  # disabled: blocks the script when run as a .py (designed for Jupyter)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, threshold_mgoe=60.0, out_path="confusion_matrix.png"):
    y_true_cls = (np.array(y_true) >= threshold_mgoe).astype(int)
    y_pred_cls = (np.array(y_pred) >= threshold_mgoe).astype(int)
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
    class_names = [f"< {threshold_mgoe} MGOe", f"≥ {threshold_mgoe} MGOe"]
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(f"CGCNN Confusion Matrix (SMOKE TEST)")
    
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    # plt.show()  # disabled: blocks the script when run as a .py (designed for Jupyter)
    plt.close()

def evaluate(model, loader):
    model.eval()
    preds, targets, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            bhmax_pred, _ = model(batch)
            preds.extend(bhmax_pred.cpu().numpy().flatten())
            targets.extend(batch.y.cpu().numpy().flatten())
            for d in batch.to_data_list(): ids.append(getattr(d, "mat_id", ""))
    
    t_mgoe, p_mgoe = np.array(targets) / MGOe_TO_Jm3, np.array(preds) / MGOe_TO_Jm3
    mask = t_mgoe >= (MAGNITO_THRESHOLD / MGOe_TO_Jm3)
    recall = float(np.mean(p_mgoe[mask] >= (MAGNITO_THRESHOLD / MGOe_TO_Jm3))) if mask.any() else float("nan")
    return mean_absolute_error(t_mgoe, p_mgoe), np.sqrt(mean_squared_error(t_mgoe, p_mgoe)), recall, t_mgoe.tolist(), p_mgoe.tolist(), ids


# ── TRAINING & INFERENCE ─────────────────────────────────────────────────────
def training_loop(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, patience, stage_name):
    best_val_mae, patience_count = float("inf"), 0
    print(f"\n  [{stage_name}] Starting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            bhmax_pred, site_pred = model(batch)
            loss = criterion(bhmax_pred, batch.y, site_pred.squeeze(-1), batch.site_m_norm, batch.has_site_labels, batch.batch)
            loss.backward()
            total_loss += loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        val_mae, _, recall, _, _, _ = evaluate(model, val_loader)
        scheduler.step(val_mae)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{epochs} | Loss: {avg_loss:.4e} | Val MAE: {val_mae:.4f} MGOe | Recall: {recall:.2%}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "best_model_temp.pt")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience: 
                print(f"  Early stopping triggered at epoch {epoch}.")
                break
    model.load_state_dict(torch.load("best_model_temp.pt", map_location=DEVICE))
    return best_val_mae

def screen_gnome_materials(model, gnome_cache_dir):
    ds = LazyGraphDataset(gnome_cache_dir, labelled_only=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    
    print(f"  Running GNoME Inference ({MC_PASSES} MC passes)...")
    model.eval()
    for _m in model.modules():            # MC dropout: keep Dropout layers stochastic while BatchNorm stays in eval
        if isinstance(_m, nn.Dropout):
            _m.train()
    all_means, all_stds, all_ids, all_dom = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            mc_u, mc_dom = [], []
            for _ in range(MC_PASSES):
                u_p, _, dom = model(batch, mc_dropout=True, return_dominance=True)
                mc_u.append(u_p.squeeze(-1))
                mc_dom.append(dom)

            mc_u = torch.stack(mc_u, dim=0).cpu().numpy()
            all_means.extend(mc_u.mean(axis=0).tolist())
            all_stds.extend(mc_u.std(axis=0).tolist())
            all_dom.extend(torch.stack(mc_dom, dim=0).cpu().numpy().mean(axis=0).tolist())
            for d in batch.to_data_list(): all_ids.append(getattr(d, "mat_id", "unknown"))

    results = []
    for mid, mean, std, dom in zip(all_ids, all_means, all_stds, all_dom):
        mean_mgoe = mean / MGOe_TO_Jm3
        std_mgoe = std / MGOe_TO_Jm3
        ci_lower_mgoe = mean_mgoe - (1.96 * std_mgoe)
        
        results.append({
            "gnome_id": mid,
            "BH_max_mean_MGOe": mean_mgoe,
            "BH_max_std_MGOe": std_mgoe,
            "BH_max_CI_lower_95_MGOe": ci_lower_mgoe,
            "high_confidence_target": bool(ci_lower_mgoe >= 60.0),
            "re_dominance_score": float(dom),
            "anisotropy_dominated": bool(dom >= 0.6)
        })
    return sorted(results, key=lambda r: r["BH_max_mean_MGOe"], reverse=True)


# ── MAIN PIPELINE ────────────────────────────────────────────────────────────
def run_cgcnn_pipeline():
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    os.makedirs(GNOME_CACHE_DIR, exist_ok=True)

    print("── 1. Graph Construction (MP & GNoME) ───────────────")
    mp_cache_file = "mp_docs_re_stable_v6.pkl"
    docs = []
    
    if os.path.exists(mp_cache_file):
        with open(mp_cache_file, "rb") as f: docs = pickle.load(f)
        
    if not docs:
        print("  Fetching MP docs (this will take a moment)...")
        RE_ELEMENTS = ["Sc","Y","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"]
        with MPRester(MP_API_KEY) as mpr:
            for sym in RE_ELEMENTS:
                try:
                    batch = mpr.materials.summary.search(
                        elements=[sym],
                        energy_above_hull=(0, 0.05),
                        fields=["structure", "total_magnetization_normalized_vol", "total_magnetization", "volume", "material_id"]
                    )
                    for doc in batch:
                        docs.append({
                            "structure": doc.structure,
                            "total_magnetization_normalized_vol": getattr(doc, "total_magnetization_normalized_vol", None),
                            "total_magnetization": getattr(doc, "total_magnetization", None),
                            "volume": getattr(doc, "volume", None),
                            "material_id": str(doc.material_id)
                        })
                except Exception as e:
                    pass
        with open(mp_cache_file, "wb") as f: pickle.dump(docs, f)

    if not docs:
        print("  CRITICAL ERROR: No MP data retrieved. Please verify your API Key.")
        sys.exit(1)

    for d in docs:
        tot_mag = d.get("total_magnetization_normalized_vol")
        if tot_mag is None:
            tot = d.get("total_magnetization")
            vol = d.get("volume")
            if tot is not None and vol is not None and vol > 0:
                tot_mag = tot / vol

        struct = d.get("structure")
        site_moms = struct.site_properties.get("magmom", None) if hasattr(struct, "site_properties") else None
        _build_and_cache_one(struct, tot_mag, site_moms, d.get("material_id"), GRAPH_CACHE_DIR)
    
    _write_index(GRAPH_CACHE_DIR)

    # Read recursively from nested extracted folders
    if not os.path.exists(os.path.join(GNOME_CACHE_DIR, "index.json")):
        if not os.path.exists(GNOME_MASTER_DIR):
            print(f"  CRITICAL ERROR: Could not find '{GNOME_MASTER_DIR}' directory.")
            sys.exit(1)
            
        print(f"  Building GNoME graphs recursively from {GNOME_MASTER_DIR}...")
        
        # glob.glob with recursive=True drills into all sub-folders automatically
        search_pattern = os.path.join(GNOME_MASTER_DIR, "**", "*.cif")
        cif_files = glob.glob(search_pattern, recursive=True)
        
        print(f"  Found {len(cif_files)} CIF files across all batches. Parsing structures...")
        
        for filename in cif_files:
            try:
                struct = Structure.from_file(filename)
                mid = os.path.splitext(os.path.basename(filename))[0]
                _build_and_cache_one(struct, None, None, mid, GNOME_CACHE_DIR)
            except Exception: 
                pass
        _write_index(GNOME_CACHE_DIR)

    print("\n── 2. Training CGCNN Backbone (80/20 Split) ─────────")
    full_dataset = LazyGraphDataset(GRAPH_CACHE_DIR, labelled_only=True)
    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.20, random_state=42)
    
    train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(torch.utils.data.Subset(full_dataset, val_idx), batch_size=BATCH_SIZE)

    model = MagGNoME().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    training_loop(model, train_loader, val_loader, optimizer, CombinedMagLoss(), scheduler, EPOCHS, PATIENCE, "Split Validation Run")

    print("\n── 3. Validation & Plotting ───────────────────────────")
    _, _, _, val_targets, val_preds, val_ids = evaluate(model, val_loader)
    
    val_df = pd.DataFrame({
        "material_id": val_ids,
        "true_BH_max_MGOe": val_targets,
        "pred_BH_max_MGOe": val_preds
    })
    val_df.to_csv("validation_predictions_v6.csv", index=False)
    print("  Saved validation performance vectors to 'validation_predictions_v6.csv'")
    
    plot_parity(val_targets, val_preds)
    plot_confusion_matrix(val_targets, val_preds, threshold_mgoe=60.0)

    print("\n── 4. Retraining on Full Dataset ──────────────────────")
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    full_model = MagGNoME().to(DEVICE)
    full_optimizer = torch.optim.Adam(full_model.parameters(), lr=LR)
    full_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(full_optimizer, mode="min", factor=0.5, patience=5)
    training_loop(full_model, full_loader, val_loader, full_optimizer, CombinedMagLoss(), full_scheduler, EPOCHS, PATIENCE, "Final Production Run")
    torch.save(full_model.state_dict(), "final_cgcnn_model.pt")

    print("\n── 5. Screening GNoME Database ────────────────────────")
    results = screen_gnome_materials(full_model, GNOME_CACHE_DIR)
    
    print("\n  Top 20 Predicted (BH)max Candidates (SMOKE TEST):")
    print(f"  {'GNoME ID':<25} {'Mean':>8} {'±σ':>8} {'95% Lower':>10} {'RE dom':>7} {'Conf':>5}")
    print(f"  {'':25} {'(MGOe)':>8} {'(MGOe)':>8} {'(MGOe)':>10}")
    print("  " + "-" * 78)
    for r in results[:20]:
        conf = "HIGH" if r["high_confidence_target"] else ("MED" if r["BH_max_mean_MGOe"] >= 60.0 else "LOW")
        print(f"  {r['gnome_id']:<25} {r['BH_max_mean_MGOe']:>8.2f} {r['BH_max_std_MGOe']:>8.2f} {r['BH_max_CI_lower_95_MGOe']:>10.2f} {r['re_dominance_score']:>7.3f} {conf:>5}")

    df_results = pd.DataFrame(results)
    df_results.to_csv("gnome_ranked_candidates_v6.csv", index=False)
    print("\nPipeline Complete. Standalone dataframe exported to 'gnome_ranked_candidates_v6.csv'.")

if __name__ == "__main__":
    run_cgcnn_pipeline()
    