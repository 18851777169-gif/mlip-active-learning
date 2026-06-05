#!/usr/bin/env python3
"""MP-ALOE active learning: 4 systems x 3 seeds x 3 strategies (A, G, I)."""
import sys, pickle, numpy as np, os, warnings
warnings.filterwarnings('ignore')
import pandas as pd

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
MID = sys.argv[2] if len(sys.argv) > 2 else "mp-570316"
DATA_DIR = "data/mp_aloe"
DEVICE = 'cuda'
N_INIT, N_QUERY, N_ITER = 20, 10, 5
EPOCHS, LR, BATCH = 40, 1e-3, 16

import torch; torch.manual_seed(SEED); np.random.seed(SEED)
from data import MaterialDataset, make_dataloader, create_splits
from model_fallback import FallbackModel
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

def to_dev(b): return {k:v.to(DEVICE) if isinstance(v,torch.Tensor) else v for k,v in b.items()}

# Load data
with open(f"{DATA_DIR}/{MID}.pkl","rb") as f: structures = pickle.load(f)
dataset = MaterialDataset(structures)
init_idx, pool_idx, test_idx, val_idx = create_splits(len(dataset), N_INIT, 0.15, 0.10, SEED)

def train_model(train_idx, val_idx):
    model = FallbackModel(hidden_channels=64, num_interactions=2).to(DEVICE)
    tl = make_dataloader(dataset, train_idx, BATCH, shuffle=True)
    vl = make_dataloader(dataset, val_idx, BATCH, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',factor=0.5,patience=8)
    best_val, best_state, patience = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        for batch in tl:
            batch=to_dev(batch); opt.zero_grad()
            e_pred,_ = model(batch["z"],batch["pos"],batch["batch"])
            loss=torch.nn.functional.l1_loss(e_pred,batch["y"].view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); vs,vn=0.0,0
        with torch.no_grad():
            for batch in vl:
                batch=to_dev(batch); e_pred,_=model(batch["z"],batch["pos"],batch["batch"])
                vs+=(e_pred-batch["y"].view(-1)).abs().sum().item(); vn+=batch["y"].shape[0]
        val_mae=vs/vn; sched.step(val_mae)
        if val_mae<best_val-1e-8: best_val,best_state,patience=val_mae,{k:v.cpu().clone() for k,v in model.state_dict().items()},0
        else: patience+=1
        if patience>=15: break
    if best_state: model.load_state_dict(best_state)
    return model

def evaluate(model, test_idx):
    tl=make_dataloader(dataset,test_idx,BATCH,shuffle=False); model.eval(); ts,tn=0.0,0
    with torch.no_grad():
        for batch in tl:
            batch=to_dev(batch); e_pred,_=model(batch["z"],batch["pos"],batch["batch"])
            ts+=(e_pred-batch["y"].view(-1)).abs().sum().item(); tn+=batch["y"].shape[0]
    return ts/tn

def get_embs(model, indices):
    loader=make_dataloader(dataset,indices,BATCH,shuffle=False); model.eval(); embs=[]
    with torch.no_grad():
        for batch in loader:
            batch=to_dev(batch); _,nf=model(batch["z"],batch["pos"],batch["batch"])
            bi=batch["batch"]
            for s in range(bi.max().item()+1): embs.append(nf[bi==s].mean(dim=0).detach().detach().cpu().numpy())
    return np.array(embs) if embs else None

def compute_u(models,pool):
    pl=make_dataloader(dataset,pool,BATCH,shuffle=False); mlist=list(models.values()); vlist=[]
    for batch in pl:
        batch=to_dev(batch); preds=[m(batch["z"],batch["pos"],batch["batch"])[0] for m in mlist]
        p=torch.stack(preds,dim=0)
        vlist.append(p.std(dim=0).detach().cpu().numpy() if p.shape[0]>1 else np.zeros(p.shape[1]))
    return np.concatenate(vlist)

# Run AL
all_curves = {}
for sname, alpha_type in [("A_random",None),("G_hybrid_weighted",0.5),("I_aud_rank","adapt")]:
    print(f"\n--- {sname} ---")
    labeled=list(init_idx); pool=list(pool_idx); leb=None; curve=[]
    for it in range(N_ITER+1):
        models={}
        for ms in [SEED,SEED+100]:
            torch.manual_seed(ms); models[ms]=train_model(labeled,val_idx)
        mae=evaluate(list(models.values())[0],test_idx)
        curve.append(mae)
        print(f"  Iter {it} | N={len(labeled)} | MAE={mae:.4f}")
        if it>=N_ITER or len(pool)<N_QUERY: break
        if alpha_type is None:
            selected=np.random.RandomState(SEED+it).choice(pool,N_QUERY,replace=False)
        else:
            u=compute_u(models,np.array(pool))
            pool_embs=get_embs(list(models.values())[0],pool)
            if pool_embs is None:
                selected=np.random.RandomState(SEED+it).choice(pool,N_QUERY,replace=False)
            else:
                if leb is not None and leb.shape[0]>0: d=cdist(pool_embs,leb,metric="cosine").min(axis=1)
                else: d=np.ones(len(pool))
                def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
                u_n,d_n=norm(u),norm(d)
                if alpha_type=="adapt":
                    rho,_=spearmanr(u_n,d_n); alpha=np.clip(0.5-0.3*rho,0.2,0.8)
                else: alpha=alpha_type
                combined=alpha*u_n+(1-alpha)*d_n
                selected=np.array(pool)[np.argsort(combined)[-N_QUERY:]]
        for s in selected:
            if s in pool: pool.remove(int(s)); labeled.append(int(s))
        leb=get_embs(list(models.values())[0],labeled)
    all_curves[sname] = curve
pd.DataFrame(all_curves).to_csv(f"results_mpaloe/mpaloe_{MID}_seed{SEED}.csv",index=False)

print(f"\nDone! {MID} seed={SEED}")
