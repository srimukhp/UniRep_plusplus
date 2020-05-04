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
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--stop', type=int, required=True)
parser.add_argument('--rnn_size', type=int, required=True)
parser.add_argument('--n_layers', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)

args = parser.parse_args()
print(args)
# UniRep parameters to set

rnn_size = args.rnn_size
n_layers = args.n_layers
initialize_mLSTM_from_scratch = False 
initialize_toplayer_from_scratch = True # does not matter for babbler unless doing Evotuning
batch_size = args.batch_size

init_path = '{}_weights/'.format(rnn_size)
model_path = '../UniRep/'+init_path
len_unirep = rnn_size

print('Loading weights from ', model_path)
save_dir = 'hs{}_nl{}_{}_to_{}'.format(rnn_size, args.n_layers, args.start, args.stop)
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
        lst = lst.reshape((-1,1))
        aa_reps[key] = lst
    len_aa_rep = len(aa_reps['M'])
else:
    raise ValueError('wtf')
len_rep = len_unirep+len_aa_rep

b_trained = babbler(batch_size=batch_size, model_path=model_path, trained=(not initialize_toplayer_from_scratch),
                   rnn_size=rnn_size, n_layers=n_layers, new=initialize_mLSTM_from_scratch)

def make_xy(row, batch_size):
    seq = row['Sequence']
    act_site = row['Active site residue']
    act_site_type = np.array(row['Active site type'])
    avg_hidden, final_hidden, final_cell = b_trained.get_reps(seq, batch_size)
    mol_rep = np.zeros((len_aa_rep, batch_size))
    count=0
    for i in act_site.index.tolist():
        mol_rep[:,count:count+1] = aa_reps[act_site[i]]
        count+=1
    final_hidden = final_hidden.T
    act_site_type = act_site_type-1
    act_site_type = act_site_type.reshape(act_site_type.shape[0],1)
    return np.concatenate((final_hidden, mol_rep), axis=0), act_site_type.T

start = args.start 
stop = args.stop

df_active_site = pd.read_excel(args.filename ,header=[0],index_col=[0])
df_active_site = df_active_site[start:stop]

num_batches = math.floor((stop - start)/batch_size)
print('num_batches: ', num_batches)

X_train = np.zeros((len_rep, num_batches*batch_size))
Y_train = np.zeros((1, num_batches*batch_size))

for i in range(num_batches):
    if (i % 10 ==0):
        print('Writing batch number :', i)
        print('X_train shape: ', X_train.shape)
        print('Y_train shape: ', Y_train.shape)
        np_concat = np.concatenate((X_train, Y_train), axis=0)

        print('Saving to ', save_dir)
        np.save(save_dir, np_concat)
    
    X_train[:,i*batch_size:(i+1)*batch_size], Y_train[:,i*batch_size:(i+1)*batch_size ] = make_xy(df_active_site.iloc[i*batch_size:(i+1)*batch_size], batch_size)


print('X_train shape: ', X_train.shape)
print('Y_train shape: ', Y_train.shape)

np_concat = np.concatenate((X_train, Y_train), axis=0)

print('Saving to ', save_dir)
np.save(save_dir, np_concat)

print('Done saving to file successfully')