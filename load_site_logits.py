import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import re
from UniRep import unirep
import time

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
	babbler=unirep.babbler1900
	batch_size = 64

	b_trained = babbler(batch_size=batch_size, model_path=f'./1000_iters_hs1900_nl4_pt/', trained=True,
					   rnn_size=1900, n_layers=1, new=False)
	df = pd.read_csv('uniprot.tab', sep='\t', lineterminator='\n')
	df = df.fillna('')
	df.loc[:, 'Active site'] = df.loc[:, 'Active site'].apply(get_sites)
	df.loc[:, 'Binding site'] = df.loc[:, 'Binding site'].apply(get_binding_sites)


	site_logits_list = []
	seqs = df['Sequence']
	print("Starting to load site logits with a batch size of", batch_size)
	for i in range(0, 12800, batch_size):
		time1=time.time()
		int_seqs = []
		for seq in seqs[i:min(i+batch_size, len(seqs))]:
			int_seqs.append(unirep.aa_seq_to_int(seq.strip())[:-1])
		n = len(int_seqs)
		lengths = np.array([len(int_seq) for int_seq in int_seqs])
		seq_array = np.zeros((n, lengths.max()))
		for j in range(n):
			seq_array[j, :lengths[j]] = int_seqs[j]

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			site_logits = sess.run(b_trained.site_logits,
										   feed_dict = {
								b_trained._batch_size_placeholder: n,
								b_trained._minibatch_x_placeholder: seq_array,
								b_trained._initial_state_placeholder: b_trained._zero_state
										   })

			for j in range(n):
				site_logits_list.append(site_logits[j][:lengths[j]])

		print(f"time to load site logits of sequences {i} to {i+batch_size}: {time.time()-time1}")

	with open('site_logits.pkl', 'wb') as f:
		pickle.dump(site_logits_list, f)
