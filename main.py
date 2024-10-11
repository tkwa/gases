# %%

import rdkit
from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors
from rdkit.Chem.Descriptors import rdMolDescriptors
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
import matplotlib.pyplot as plt
import pandas as pd
from thermo.group_contribution.joback import Joback
import importlib
import UManSysProp_public.umansysprop.boiling_points as bp
from openbabel import pybel
import chemicals

from multiprocessing import  Pool

# import swifter
# from swifter import set_defaults
# set_defaults(allow_dask_on_strings=False,)

# client = Client(processes=True)


directory = 'gdb11'

df = pd.DataFrame()
for i in range(1, 10):
    data_this_length = pd.read_csv(f'{directory}/gdb11_size{i:02d}.smi',
                                   sep=r"\s+", header=None)
    # set headers
    data_this_length.columns = ['smiles', 'index_in_len', 'type']
    data_this_length['length'] = i
    df = pd.concat([df, data_this_length], ignore_index=True)

# %%

def process(x):

    smiles, length = x
    molwt = float('nan')
    Tb = float('nan')
    is_gas = False

    # no gases have >7 non-F atoms
    if length - smiles.count('F') > 7:
        is_gas = False
        Tb = float('nan')
        return {'molwt': molwt, 'is_gas': is_gas, 'Tb': Tb, 'method': 'non-F atoms'}
    
    mol = rdkit.Chem.MolFromSmiles(smiles)
    molwt = Descriptors.ExactMolWt(mol)
    if molwt > 300:  # heaviest known gas is 294
        is_gas = False
        Tb = float('nan')
        return {'molwt': molwt, 'is_gas': is_gas, 'Tb': Tb, 'method': 'molwt'}
    
    try:
        cas = chemicals.identifiers.CAS_from_any(smiles, autoload=False, cache=True)
        method = 'YAWS'
        Tb = chemicals.phase_change.Tb(cas, method=method)
        is_gas = Tb < 298
        return {'molwt': molwt, 'is_gas': is_gas, 'Tb': Tb, 'method': method}
    except:
        mol = pybel.readstring("smi", smiles)
        Tb = bp.nannoolal(mol)
        is_gas = Tb < 298

    ret = {'molwt': molwt, 'is_gas': is_gas, 'Tb': Tb, 'method': 'nannoolal'}
    return ret
# %%
# Use itertuples to create an efficient iterable
iterable = df[['smiles', 'length']].itertuples(index=False, name=None)

bp_data = process_map(process, iterable, max_workers=24, chunksize=4000, total=len(df))

# Print method distribution
method_counts = pd.Series([x['method'] for x in bp_data]).value_counts()
print(method_counts)


# %%
df = pd.concat([df, pd.DataFrame(bp_data)], axis=1)
# %%
# Histogram of boiling points

df['Tb'].hist(bins=100)

# Vertical line at 298 K
plt.axvline(x=298, color='r')

plt.show()

print(f"Number of gases: {len(df[df['Tb'] < 298])} of {len(df)}")
print(f"Used Nannoolal method for {len(df.dropna())} of {len(df)}")

# row 2344
# %%

gases_only = df[df['is_gas']]

# random sample 400
gases_only = gases_only.sample(400)

ims = []
for i, row in tqdm(gases_only.iterrows()):
    print(row.smiles)
    print(f"Est. Tb={row.Tb:.2f} K")
    print(f"MW={row.molwt}")
    # draw
    mol = rdkit.Chem.MolFromSmiles(row.smiles)
    im = rdkit.Chem.Draw.MolToImage(mol, size=(100,100))
    ims.append(im)
    print()

# %%


# Put all images in a grid
n = len(ims)
rows, cols = n//20 + 1, 20
fig, axs = plt.subplots(rows, cols, figsize=(80, 80))
for i, im in enumerate(ims):
    ax = axs[i//cols, i%cols]
    ax.imshow(im)
    ax.axis('off')
    # caption with BP and MW
    ax.set_title(f"Tb={gases_only.iloc[i].Tb:.2f} K, MW={gases_only.iloc[i].molwt:.2f}")
# remove empty subplots
for i in range(n, rows*cols):
    fig.delaxes(axs.flatten()[i])
plt.tight_layout()

# export
plt.savefig('gases.png')