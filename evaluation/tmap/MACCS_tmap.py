
from distutils.file_util import write_file
import numpy as np
import pandas as pd
from regex import P
import tmap as tm
import scipy.stats as ss
from faerun import Faerun
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, MACCSkeys
from mhfp.encoder import MHFPEncoder
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

bits = 1024

MACCS_fps = []
labels = []
groups = []
mh_all = []
label_down = []
types = []
tpsa = []
logp = []
mw = []
h_acceptors = []
h_donors = []
ring_count = []
has_coc = []
has_sa = []
has_tz = []
sim=[]


df = pd.read_csv("./input.csv").dropna(subset=["smiles"]).reset_index(drop=True)
custom_cmap = ListedColormap(
    ["#1E90FF", "#eb2f06", "#fa983a", "#d9d919", "#78e08f", "#4a69bd"],
    name="custom",
)


lf = tm.LSHForest(bits)
#enc = tm.Minhash(128)
enc = tm.Minhash(bits)


substruct_coc = AllChem.MolFromSmiles("COC")
substruct_sa = AllChem.MolFromSmiles("NS(=O)=O")
substruct_tz = AllChem.MolFromSmiles("N1N=NN=C1")



total = len(df)
for i, row in df.iterrows():
    if i % 100 == 0 and i > 0:
        print(f"{round(100 * (i / total))}% done ...")

    smiles = row[0]
    mhfps=df.iloc[i,3:]
    mh=list(mhfps)
    mh_all.append(mh)
    mol = AllChem.MolFromSmiles(smiles)

    if mol and mol.GetNumAtoms() > 5 and smiles.count(".") < 2:
        fps = MACCSkeys.GenMACCSKeys(mol)
        MACCS_fps.append(tm.VectorUint(fps))
        types.append(row[1])


print("start fps")
lf.batch_add(enc.batch_from_binary_array(MACCS_fps))
lf.index()
print("fps finish!")


types_value=[]
for i, value in enumerate(types):
    if value==1:
        types_value.append(1)
    elif value==2:
        types_value.append(2)
    elif value==0:
        types_value.append(0)
    elif value==3:
        types_value.append(3)

cfg = tm.LayoutConfiguration() 
cfg.k = 100
# cfg.sl_extra_scaling_steps = 1
cfg.sl_repeats = 2
cfg.mmm_repeats = 2
cfg.node_size = 1
x,y,s,t,inf = tm.layout_from_lsh_forest(lf, config=cfg)


legend_labels = [
    (1, "generated"),
    (0, "background"),
]


faerun = Faerun(
    clear_color="#FFFFFF",
    view="front",
    coords=False,
    legend_title="Legend",
)

faerun.add_scatter(
    "tests",
    {"x": x, "y": y, "c":[types_value],"labels":df["smiles"]},
    point_scale=4.5,
    colormap = [custom_cmap],
    has_legend=True,
    series_title = ['Group'],
    categorical=[True],
    shader="smoothCircle",
    title_index=2,
    legend_title="",
    legend_labels=legend_labels,
)

faerun.add_tree(
    "tests_tree", {"from": s, "to": t}, point_helper="tests", color="#666666"
)


faerun.plot('tmap_result', template="smiles")

