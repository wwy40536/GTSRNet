import os
import math
import h5py
import numpy as np
import sys
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Config
# =========================
class CFG:
    path = '../dataset/data_git_version.h5'
    # segment length (生成完整片段长度)
    seg_len = 24         # 24小时一个片段（可改48）
    stride = 1

    # training speed: 每个 batch 抽多少栅格点（节点）
    node_sample = 1024   # 512/1024/2048 看显存
    batch_size = 32
    epochs = 5
    lr = 1e-3
    weight_decay = 1e-4
    num_workers = 0

    # diffusion
    T = 200              # diffusion steps（先用200，跑通后可加到1000）
    beta_start = 1e-4
    beta_end = 0.02

    # model
    hidden_size = 128
    num_heads = 4
    n_encoder = 2
    n_decoder = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 123


cfg = CFG()


# =========================
# Utils
# =========================
def set_seed(seed=123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def minmax_fit(x):
    # x: (T,N,C)
    dmin = x.min(axis=(0,1), keepdims=True)
    dmax = x.max(axis=(0,1), keepdims=True)
    scale = (dmax - dmin) + 1e-6
    return dmin, dmax, scale

def minmax_transform(x, stats):
    dmin, dmax, scale = stats
    return (x - dmin) / scale

def minmax_inverse(x, stats):
    dmin, dmax, scale = stats
    return x * scale + dmin


# =========================
# Dataset: sample full segments (L,3) from nodes
# =========================
class SegmentDataset(Dataset):
    """
    raw data: (T, N, C)  e.g. (1488,10000,3)
    each sample: one node's segment window -> (L, C)
    """
    def __init__(self, data_norm, seg_len=24, stride=1, node_sample=1024, seed=123):
        self.data = data_norm.astype(np.float32)
        self.T, self.N, self.C = self.data.shape
        self.seg_len = seg_len
        self.stride = stride
        self.node_sample = node_sample
        self.rng = np.random.default_rng(seed)

        self.starts = []
        max_s = self.T - seg_len
        for s in range(0, max_s + 1, stride):
            self.starts.append(s)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        # window: (L, N, C)
        w = self.data[s:s+self.seg_len]  # (L,N,C)

        # sample some nodes -> (L, n, C)
        if self.node_sample is not None and self.node_sample < self.N:
            nodes = self.rng.choice(self.N, size=self.node_sample, replace=False)
            w = w[:, nodes, :]

        # return as (n, L, C) for easy flatten later
        # each node is an independent sequence sample
        w = np.transpose(w, (1,0,2))  # (n,L,C)
        return torch.from_numpy(w)     # float32


# =========================
# Backbone: DiM (your reference) or fallback transformer
# =========================
def build_backbone(seg_len, channels):
    """
    Returns epsilon_theta(x_t, t): predicts noise, same shape as x_t
    Expected input: (B, L, C)
    """
    # try to load your uploaded DiM file as reference backbone
    # NOTE: your file depends on Models.interpretable_diffusion.SMamba
    # If missing, we fallback.
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("dim_ref", "/mnt/data/621e6352-2c25-446c-929b-ea8f6cd88af4.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        DiM = mod.DiM

        model = DiM(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            n_encoder=cfg.n_encoder,
            n_decoder=cfg.n_decoder,
            feature_last=True,
            dropout=0.1,
            input_shape=(seg_len, channels),
            trans_mx=None,
        )
        return model

    except Exception as e:
        print("[WARN] DiM import failed, fallback to TinyTransformer. Reason:", str(e))

        class TinyEpsModel(nn.Module):
            def __init__(self, L, C, d=cfg.hidden_size, heads=cfg.num_heads, layers=2):
                super().__init__()
                self.inp = nn.Linear(C, d)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d, nhead=heads, dim_feedforward=d*4,
                    dropout=0.1, batch_first=True, activation="gelu", norm_first=True
                )
                self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
                self.time_emb = nn.Embedding(cfg.T, d)
                self.out = nn.Linear(d, C)

            def forward(self, x, t):
                # x: (B,L,C) ; t: (B,)
                h = self.inp(x)
                h = h + self.time_emb(t).unsqueeze(1)
                h = self.enc(h)
                return self.out(h)  # (B,L,C)

        return TinyEpsModel(seg_len, channels)


# =========================
# Diffusion wrapper (DDPM)
# =========================
class DDPM(nn.Module):
    def __init__(self, eps_model, T=200, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.eps_model = eps_model
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise):
        """
        x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        x0: (B,L,C), t: (B,), noise: same
        """
        b = x0.size(0)
        s1 = self.sqrt_alphas_cumprod[t].view(b, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1)
        return s1 * x0 + s2 * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.eps_model(xt, t)   # predict noise
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
        """
        one reverse step: x_{t-1} from x_t
        """
        b = x.size(0)
        beta_t = self.betas[t].view(b,1,1)
        alpha_t = self.alphas[t].view(b,1,1)
        alpha_bar_t = self.alphas_cumprod[t].view(b,1,1)

        eps = self.eps_model(x, t)

        # DDPM mean
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps)

        if (t == 0).all():
            return mean
        noise = torch.randn_like(x)
        var = beta_t
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, batch_size, L, C, device):
        x = torch.randn(batch_size, L, C, device=device)
        for ti in reversed(range(self.T)):
            t = torch.full((batch_size,), ti, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x  # (B,L,C)


# =========================
# Load data
# =========================
def load_h5(path):
    with h5py.File(path, "r") as f:
        data = f["data"][:]  # (T, N, C)
        idx = f["idx"][:]
    idx = np.array([x.decode() for x in idx])
    return data.astype(np.float32), idx


# =========================
# Main: train + generate segments
# =========================
def main():
    set_seed(cfg.seed)
    print("Device:", cfg.device)

    data, time_idx = load_h5(cfg.h5_path)
    print("Raw:", data.shape, time_idx[0], "->", time_idx[-1])  # expect (1488,10000,3)

    stats = minmax_fit(data)
    data_norm = minmax_transform(data, stats)

    ds = SegmentDataset(data_norm, seg_len=cfg.seg_len, stride=cfg.stride, node_sample=cfg.node_sample, seed=cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    # Build model
    eps_model = build_backbone(cfg.seg_len, channels=3).to(cfg.device)
    ddpm = DDPM(eps_model, T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end).to(cfg.device)

    opt = torch.optim.AdamW(ddpm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Train
    ddpm.train()
    for ep in range(1, cfg.epochs+1):
        losses = []
        for w in loader:
            # w: (B, n_nodes, L, C) because dataset returns (n,L,C) and dataloader batches it
            # we flatten nodes into batch dimension -> (B*n, L, C)
            w = w.to(cfg.device)
            B, n, L, C = w.shape
            x0 = w.view(B*n, L, C)

            t = torch.randint(0, cfg.T, (x0.size(0),), device=cfg.device, dtype=torch.long)
            loss = ddpm.p_losses(x0, t)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())

        print(f"Epoch {ep:03d} | loss: {np.mean(losses):.6f}")

    # Generate segments (full sequence)
    ddpm.eval()
    gen = ddpm.sample(batch_size=64, L=cfg.seg_len, C=3, device=cfg.device).cpu().numpy()
    gen_inv = minmax_inverse(gen, stats)  # back to original scale

    os.makedirs("gen_out", exist_ok=True)
    np.save("gen_out/generated_segments.npy", gen_inv)
    print("Saved generated full segments:", gen_inv.shape, "-> gen_out/generated_segments.npy")
    print("Example stats real mean/std:", data.mean(), data.std())
    print("Example stats gen  mean/std:", gen_inv.mean(), gen_inv.std())


if __name__ == "__main__":
    main()
