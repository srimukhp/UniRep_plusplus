import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import re
from UniRep import unirep
import time
import argparse
import os


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def get_sites(string, site_type='ACT_SITE'):
	"""string=site entry
	Returns the sites as a list, e.g. if 
	string = 'ACT_SITE 85;  /evidence=...; ACT_SITE 110;  /evidence=...'
	returns [85, 110]"""
	
	pattern = re.compile(f'{site_type} [0-9]+;')
	matches = pattern.findall(string)
	get_site = lambda match: int(match.split(' ')[1][:-1])
	return [get_site(match) for match in matches]
get_binding_sites = lambda string: get_sites(string, 'BINDING')

def process_domain(string):
	pattern = re.compile('DOMAIN [0-9]+..[0-9]+')
	matches = pattern.findall(string)
	intervals = [[int(num) for num in match.split()[1].split('..')] for match in matches]

	dn_pattern = re.compile('note="[^;]*"')
	domain_names = [s.split('"')[1] for s in dn_pattern.findall(string)]

	return list(zip( domain_names, intervals))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='load reps')
	parser.add_argument('--path', type=str, default='./1000_iters_hs1900_nl4_pt/')
	parser.add_argument('--rnn_size', type=int, default=1900)
	parser.add_argument('--n_layers', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--inds', type=str, default=None)
	parser.add_argument('--num_reps', type=int, default=12800)
	parser.add_argument('--continuee', type=str2bool, default=False)

	args = parser.parse_args()
	batch_size = args.batch_size
	babbler=unirep.babbler1900

	b_trained = babbler(batch_size=batch_size, model_path=args.path, trained=True,
					   rnn_size=args.rnn_size, n_layers=args.n_layers, new=False)
	df = pd.read_csv('uniprot.tab', sep='\t', lineterminator='\n')
	df = df.fillna('')
	df.loc[:, 'Active site'] = df.loc[:, 'Active site'].apply(get_sites)
	df.loc[:, 'Binding site'] = df.loc[:, 'Binding site'].apply(get_binding_sites)

	if args.inds is None:
		inds = None
	else:
		with open(args.inds, 'rb') as f:
			inds = pickle.load(f)

	seqs = df['Sequence']
	starti = 0
	if args.continuee and os.path.exists(f"reps/{args.path}/i.txt"):
		with open(f"reps/{args.path}/i.txt", 'rb') as f:
			starti = int(f.readline())

	print("Loading reps", batch_size)
	final_cells, final_hiddens, avg_hiddens, specific_hiddens = [], [], [], []
	for i in range(starti, args.num_reps, batch_size):
		time1=time.time()
		if inds is not None:
			binds = inds[i:i+batch_size]
		else:
			binds = None
		ah, fh, fc, sh = b_trained.get_reps(seqs[i:min(i+batch_size, len(seqs))], batch_size, binds)

		if not os.path.exists(f"reps/{args.path}"):
			os.makedirs(f"reps/{args.path}")

		if i>0:
			letter = 'a'
		else:
			letter = 'w'

		with open(f"reps/{args.path}/reps.pkl", f"{letter}b") as f:
			pickle.dump((ah, fh, fc, sh), f)

		with open(f"reps/{args.path}/i.txt", 'w') as f:
			f.write(str(i+batch_size))
		

		print(f"time to get reps of sequences {i} to {i+batch_size}: {time.time()-time1}")
