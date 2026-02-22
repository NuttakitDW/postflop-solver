#!/usr/bin/env python3
"""
train_ranger_onecycle.py
----------------------------------------------------
Modern turn-value net with up-to-date training stack.

Outputs every epoch
-------------------
models/100k_ranger_onecycle/
 ├─ best_ema.pt          ← checkpoint (EMA weights + scalers)
 ├─ loss_curve.png       ← two-panel plot (log + zoom)
 └─ losses.csv           ← per-epoch train / val loss
"""

import os, csv, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- constants / paths --------
RNG_SEED = 42
K, BOARD_FEATS = 1000, 15
IN_DIM, OUT_DIM = BOARD_FEATS + 2*K, 2*K

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data",   "training_data_100k")
OUT_DIR  = os.path.join(BASE, "..", "models", "100k_ranger_onecycle")
os.makedirs(OUT_DIR, exist_ok=True)

# -------- hyper-parameters ----------
EPOCHS       = 500
BATCH        = 512
LR_MAX       = 3e-3
WEIGHT_DECAY = 1e-4
HIDDEN       = 500
LAYERS       = 7
DROPOUT      = 0.0
CLIP         = 5.0
TEST_FRAC    = 0.20
EMA_DECAY    = 0.999
EARLY_STOP   = 50          # epochs w/o val-improve

# -------- model ----------
class ZeroSum(nn.Module):
    def forward(self, raw, r_oop, r_ip):
        cfv_oop, cfv_ip = raw[:, :K], raw[:, K:]
        g = (r_oop*cfv_oop).sum(1, keepdim=True) + (r_ip*cfv_ip).sum(1, keepdim=True)
        corr = g/2
        return torch.cat([cfv_oop - corr/r_oop.sum(1, keepdim=True).clamp(1e-8),
                          cfv_ip  - corr/r_ip .sum(1, keepdim=True).clamp(1e-8)], 1)

class Net(nn.Module):
    def __init__(self, h=500, n_layers=7, drop=0.):
        super().__init__()
        seq, d = [], IN_DIM
        for _ in range(n_layers):
            seq += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            if drop: seq.append(nn.Dropout(drop))
            d = h
        seq.append(nn.Linear(h, OUT_DIM))
        self.backbone, self.zs = nn.Sequential(*seq), ZeroSum()
    def forward(self, x):
        raw = self.backbone(x)
        return self.zs(raw,
                       x[:, BOARD_FEATS:BOARD_FEATS+K],
                       x[:, BOARD_FEATS+K:])

# -------- board-group split ----------
def board_split(arr, frac=0.2, seed=42):
    keys = np.round(arr[:, :12], 4)
    ids  = {tuple(k): i for i,k in enumerate({tuple(r) for r in keys})}
    idx  = np.array([ids[tuple(k)] for k in keys])
    rng  = np.random.RandomState(seed)
    tst  = rng.choice(len(ids), int(len(ids)*frac), replace=False)
    mask = np.isin(idx, tst)
    return np.where(~mask)[0], np.where(mask)[0]

# -------- plotting helper ----------
def save_loss_plot(train_hist, test_hist):
    skip = 5
    if len(train_hist) <= skip:
        return ""
    ep = np.arange(skip+1, len(train_hist)+1)
    tr, te = train_hist[skip:], test_hist[skip:]

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4))
    ax1.plot(ep,tr,label="Train"); ax1.plot(ep,te,label="Test")
    ax1.set_yscale("log"); ax1.set_title("Full (log-y)")
    ax1.grid(alpha=.3); ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE")

    ax2.plot(ep,tr,label="Train"); ax2.plot(ep,te,label="Test")
    allv=np.concatenate([tr,te]); lo,hi=allv.min(),np.percentile(allv,95)
    ax2.set_ylim(lo-.05*(hi-lo), hi+.05*(hi-lo))
    ax2.set_title("Zoom (≤95 pct)"); ax2.grid(alpha=.3)
    for ax in (ax1,ax2): ax.legend()

    plt.tight_layout()
    path=os.path.join(OUT_DIR,"loss_curve.png")
    plt.savefig(path,dpi=150); plt.close()
    return path

# -------- training ----------
def main():
    torch.manual_seed(RNG_SEED); np.random.seed(RNG_SEED)
    if torch.backends.mps.is_available():
        dev=torch.device("mps")
    elif torch.cuda.is_available():
        dev=torch.device("cuda")
    else:
        dev=torch.device("cpu")
    print("Device:",dev)

    # load --------------------------------------------------------------
    x=np.load(os.path.join(DATA_DIR,"inputs.npy")).astype(np.float32)
    y=np.load(os.path.join(DATA_DIR,"targets.npy")).astype(np.float32)
    tr,te=board_split(x,TEST_FRAC,RNG_SEED)

    # range-safe scaling -----------------------------------------------
    mu, std = x[tr,:BOARD_FEATS].mean(0), x[tr,:BOARD_FEATS].std(0)+1e-8
    x[:,:BOARD_FEATS]=(x[:,:BOARD_FEATS]-mu)/std
    y_scale=np.abs(y[tr]).max()+1e-8; y/=y_scale

    tl=DataLoader(TensorDataset(torch.tensor(x[tr]), torch.tensor(y[tr])),
                  batch_size=BATCH, shuffle=True)
    vl=DataLoader(TensorDataset(torch.tensor(x[te]), torch.tensor(y[te])),
                  batch_size=BATCH)

    net=Net(HIDDEN,LAYERS,DROPOUT).to(dev)

    # Ranger21 optimiser -----------------------------------------------
    from ranger21 import Ranger21              # pip install ranger21
    opt=Ranger21(net.parameters(),
                 lr=LR_MAX, weight_decay=WEIGHT_DECAY,
                 num_epochs=EPOCHS,
                 num_batches_per_epoch=len(tl))

    # One-Cycle LR ------------------------------------------------------
    tot_steps=len(tl)*EPOCHS
    sched=torch.optim.lr_scheduler.OneCycleLR(
        opt,max_lr=LR_MAX,total_steps=tot_steps,
        pct_start=0.1,anneal_strategy="cos",
        cycle_momentum=False,div_factor=10,final_div_factor=1e4
    )
    loss_fn=nn.MSELoss()

    # EMA ---------------------------------------------------------------
    ema={n:p.clone().detach() for n,p in net.named_parameters() if p.requires_grad}

    train_hist,test_hist=[],[]
    best,wait=1e9,0
    for ep in range(1,EPOCHS+1):
        net.train(); tot=n=0
        for xb,yb in tqdm(tl,leave=False,desc=f"E{ep:03d}"):
            xb,yb=xb.to(dev),yb.to(dev)
            loss=loss_fn(net(xb),yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),CLIP)
            opt.step(); sched.step()

            with torch.no_grad():
                for name,p in net.named_parameters():
                    if p.requires_grad:
                        ema[name].mul_(EMA_DECAY).add_(p,alpha=1-EMA_DECAY)
            tot+=loss.item()*xb.size(0); n+=xb.size(0)
        tr_loss=tot/n

        # ----- validation w/ EMA weights ------------------------------
        backup={}
        with torch.no_grad():
            for name,p in net.named_parameters():
                if p.requires_grad:
                    backup[name]=p.data.clone()
                    p.data.copy_(ema[name])

        net.eval(); tot=n=0
        with torch.no_grad():
            for xb,yb in vl:
                tot+=loss_fn(net(xb.to(dev)),yb.to(dev)).item()*xb.size(0)
                n  +=xb.size(0)
        te_loss=tot/n

        with torch.no_grad():
            for name,p in net.named_parameters():
                if p.requires_grad:
                    p.data.copy_(backup[name])

        train_hist.append(tr_loss); test_hist.append(te_loss)
        img_path=save_loss_plot(train_hist,test_hist)

        if te_loss<best-1e-6:
            best,wait=te_loss,0
            torch.save({**ema,"_mu":mu,"_std":std,"_yscale":y_scale},
                       os.path.join(OUT_DIR,"best_ema.pt"))
        else:
            wait+=1

        print(f"Epoch {ep:3d}  train={tr_loss:.6f}  test={te_loss:.6f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  best={best:.6f}")

        if wait>=EARLY_STOP:
            print(f"Early stop: no val gain in {EARLY_STOP} epochs."); break

    # CSV ---------------------------------------------------------------
    with open(os.path.join(OUT_DIR,"losses.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","train","test"])
        for i,(tr,te) in enumerate(zip(train_hist,test_hist),1):
            w.writerow([i,f"{tr:.8f}",f"{te:.8f}"])
    print("Loss curve →",img_path)

if __name__=="__main__":
    main()
