# Generates a vector represention for each amino acid 
# Uses the additional RDKit features in the D-MPNN paper (J. Chem. Inf. Model. 2019, 59, 8, 3370-3388)
# Each vector representation is has a length of 154; The D-MPNN features uses 200 features, but only
# 154 of them are relevent for our case

# Dependencies:
#   descriptastorus (https://github.com/bp-kelley/descriptastorus)
#   RDKit (https://rdkit.org/docs/Install.html)

from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from descriptastorus.descriptors import rdNormalizedDescriptors
from collections import OrderedDict
import numpy as np
from rdkit import Chem
import json
# generator = MakeGenerator(("RDKit2D",))
generator = rdNormalizedDescriptors.RDKit2DNormalized()

def rdkit_2d_normalized_features(aa: str):
    mol = Chem.rdmolfiles.MolFromSequence(aa)
    if aa == 'U': 
        smiles = 'C([C@@H](C(=O)O)N)[Se]'
    elif aa == 'O':
        smiles = 'C[C@@H]1CC=N[C@H]1C(=O)NCCCC[C@@H](C(=O)O)N'
    elif aa == 'X':
        return np.zeros((200,))
    elif aa == 'Z':
        smiles = 'C(CC(=O)O)[C@@H](C(=O)O)N'
    elif aa == 'B':
        smiles = 'C([C@@H](C(=O)O)N)C(=O)O'
    elif aa == 'J':
        return (np.array(rdkit_2d_normalized_features('L')) +np.array( rdkit_2d_normalized_features('I'))) / 2.0
    else:
        smiles = Chem.MolToSmiles(mol)
    # n.b. the first element is true/false if the descriptors were properly computed
    results = generator.process(smiles)
    processed, features = results[0], results[1:]
    if processed is None:
        print('Failed to compute: ',aa)
    # if processed is None, the features are are default values for the type
    return features

# Lookup tables from UniRep
aa_to_int = {
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10,
    'C':11,
    'U':12,
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21,
    'O':22, #Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or GLutamine
    'B':23, # Asparagine or aspartic acid
    'J':23, # Leucine or isoleucine
}

aa_to_int = OrderedDict(aa_to_int.items())

aa_list = list(aa_to_int.keys())
num_aa = len(aa_list)
np_rdkit_aa_reps = np.zeros((200,num_aa))

for idx,aa in enumerate(aa_list):
#     print('aa: ',aa)
    lst = rdkit_2d_normalized_features(aa)
    len_lst = len(lst)
    np_rdkit_aa_reps[:,idx] = np.array(lst)
    
np_rdkit_aa_reps[:,22] = np.mean(np_rdkit_aa_reps, axis=1)

mean = np.mean(np_rdkit_aa_reps, axis=1)
std_dev = np.std(np_rdkit_aa_reps, axis=1)

isvalid = np.logical_and(np.logical_not(np.isnan(mean)), (std_dev > 1E-6))

valid_features = np.where(isvalid == True)[0]
np_rdkit_aa_reps_valid = np_rdkit_aa_reps[valid_features,:]

aa_to_rep = OrderedDict()
for idx,key in enumerate(aa_to_int):
#     print(key)
    aa_to_rep[key] = list(np_rdkit_aa_reps_valid[:,idx])

print('Writing RDKit representations to RDKit_reps.json')

with open('RDKit_reps.json','w') as f:
    f.truncate()
    f.write(json.dumps(aa_to_rep))