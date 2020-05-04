import tensorflow as tf
from tensorflow.python.framework import ops

import os
import sys

import numpy as np
import pandas as pd
import json
import math
import argparse
import sys

new_path = os.getcwd().replace('/UniProt_data','')
print('new_path: ',new_path)
sys.path.insert(1, new_path)
from UniRep.unirep import babbler1900 as babbler


parser = argparse.ArgumentParser(description='Making UniRep+aa_rep for amino acid squences')
parser.add_argument('--filename', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--stop', type=int, required=True)
parser.add_argument('--init_path', type=str, required=True)
parser.add_argument('--rnn_size', type=int, required=True)
parser.add_argument('--n_layers', type=int, required=True)

args = parser.parse_args()
print(args)
# UniRep parameters to set

rnn_size = args.rnn_size
n_layers = args.n_layers
initialize_mLSTM_from_scratch = False 
initialize_toplayer_from_scratch = True # does not matter for babbler unless doing Evotuning
batch_size = 1
model_path = '../UniRep/'+args.init_path
len_unirep = rnn_size

print('Loading weights from ', model_path)
# Amino acid representations

use_dmpnn = True
num_class = 5

if use_dmpnn:
    with open('../D-MPNN_reps.json','r') as f:
        aa_reps = json.loads(f.read())
    for key, val in aa_reps.items():
        lst = aa_reps[key][1:-1].split(',')
        lst = [float(x) for x in lst]
        lst = np.array(lst)
        aa_reps[key] = lst
    len_aa_rep = len(aa_reps['M'])
else:
    raise ValueError('wtf')
len_rep = len_unirep+len_aa_rep

b_trained = babbler(batch_size=batch_size, model_path=model_path, trained=(not initialize_toplayer_from_scratch),
                   rnn_size=rnn_size, n_layers=n_layers, new=initialize_mLSTM_from_scratch)

def make_xy(row):
    seq = row['Sequence']
    act_site = row['Active site residue']
    act_site_type = row['Active site type']
    avg_hidden, final_hidden, final_cell = b_trained.get_rep(seq)
    mol_rep = aa_reps[act_site]
    return np.concatenate((np.reshape(final_hidden, (-1,)), mol_rep)), int(act_site_type-1)

start = args.start 
stop = args.stop

df_active_site = pd.read_excel(args.filename ,header=[0],index_col=[0])
df_active_site = df_active_site[start:stop]

m = stop - start
X_train = np.zeros((len_rep, m))
Y_train = np.zeros((m,1))

for row in range(m):
    if (row % 10 ==0):
        print('Writing row number :', row)
        np_concat = np.concatenate((X_train, Y_train.T), axis=0)

        print('Saving to ', args.save_dir)
        np.save(args.save_dir, np_concat)

    X_train[:,row], Y_train[row] = make_xy(df_active_site.iloc[row])


print('X_train shape: ', X_train.shape)
print('Y_train shape: ', Y_train.shape)

np_concat = np.concatenate((X_train, Y_train.T), axis=0)

print('Saving to ', args.save_dir)
np.save(args.save_dir, np_concat)

print('Done saving to file successfully')