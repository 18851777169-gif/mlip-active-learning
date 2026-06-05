import pickle, numpy as np
from mace.calculators import MACECalculator
calc = MACECalculator(model_path='/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model', device='cuda', default_dtype='float32')
for mid in ['mp-570316','mp-22046','mp-729184','mp-632401']:
    with open(f'data/mp_aloe/{mid}.pkl','rb') as f: structures = pickle.load(f)
    print(f'Labeling {mid} ({len(structures)} structures)...')
    for s in structures:
        s.calc = calc
        try: s.info['energy'] = s.get_potential_energy()
        except: s.info['energy'] = 0
    with open(f'data/mp_aloe/{mid}.pkl','wb') as f: pickle.dump(structures, f)
    print(f'  E=[{min(s.info["energy"] for s in structures):.1f}, {max(s.info["energy"] for s in structures):.1f}]')
print('Done!')
