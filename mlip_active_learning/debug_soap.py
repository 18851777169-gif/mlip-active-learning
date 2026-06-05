import pickle, numpy as np
from dscribe.descriptors import SOAP
S=[1,6,8,12,13,14,26,27,28,29,40,78]
soap=SOAP(species=S, periodic=True, r_cut=5.0, n_max=4, l_max=3, average="inner", sparse=False)
with open("data/ms25_labeled/FeNiCrCoCu_HEA.pkl","rb") as f: structs=pickle.load(f)
r=soap.create(structs[:5])
print(f"type={type(r)}, len={len(r)}")
for i,x in enumerate(r): 
    print(f"  [{i}]: type={type(x)}, shape={x.shape}")
