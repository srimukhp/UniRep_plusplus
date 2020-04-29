import numpy as np
import pandas as pd
import re
import tensorflow as tf
import argparse
import os
from UniRep.unirep import babbler1900 as babbler
import matplotlib.pyplot as plt

def get_sites(string, site_type='ACT_SITE'):
    """string=site entry
    Returns the sites as a list, e.g. if 
    string = 'ACT_SITE 85;  /evidence=...; ACT_SITE 110;  /evidence=...'
    returns [85, 110]"""
    
    pattern = re.compile(f'{site_type} [0-9]+;')
    matches = pattern.findall(string)
    get_site = lambda match: int(match.split(' ')[1][:-1])
    return [get_site(match) for match in matches]

def nonpad_len(batch):
    nonzero = batch > 0
    lengths = np.sum(nonzero, axis=1)/2
    return lengths

get_binding_sites = lambda string: get_sites(string, 'BINDING')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train unirep')
    parser.add_argument('--num_iters', type=int, default=100)
    parser.add_argument('--init_path', type=str, default='.')
    parser.add_argument('--rnn_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--site_loss', type=bool, default=False)

    args = parser.parse_args()
    print(args)

    if not os.path.exists('processed_uniprot.csv'):
        print("processing uniprot data and saving to csv for future reference")
        df = pd.read_csv('uniprot.tab', sep='\t', lineterminator='\n')
        df = df.fillna('')
        df.loc[:, 'Active site'] = df.loc[:, 'Active site'].apply(get_sites)
        df.loc[:, 'Binding site'] = df.loc[:, 'Binding site'].apply(get_binding_sites)
        sequences = []
        for i,row in df.iterrows():
            seq_len = row['Length']
            site_seq = np.zeros(seq_len+1,dtype=np.int32)
            site_seq[np.array(row['Active site'], dtype=np.int32)] = 1
            site_seq[np.array(row['Binding site'], dtype=np.int32)] = 2
            sequences.append(site_seq)
        df['Site sequence'] = sequences
        df.to_csv('processed_uniprot.csv')
    else:
        print("Uniprot data had already been processed. Loading from csv")
        df = pd.read_csv('processed_uniprot.csv')

    batch_size = 12
    b = babbler(batch_size=batch_size, model_path=args.init_path, trained=args.init_path != '.', 
                rnn_size=args.rnn_size, n_layers=args.n_layers, new=args.init_path == '.')

    df_le = df[df['Length']<400]
    test_mask = (np.arange(len(df_le)) % 5)==0
    df_tr = df_le[~test_mask]
    df_test = df_le[test_mask]
    print(f"Size of training data: {len(df_tr)}, size of test data: {len(df_test)}")

    with open("formatted.txt", "w") as destination:
        for i,row in df_tr.iterrows():
            seq = row['Sequence']
            site_seq = np.array(row['Site sequence'][1:-1].replace('\n', '').split(' '), dtype=np.int32)
            if b.is_valid_seq(seq): 
                seq = b.format_seq(seq)
                interleaved_seq = np.zeros(2*len(seq), dtype=np.int32)
                # interleave the amino acid sequence and site sequence
                # because this is how we are storing both datas
                interleaved_seq[::2] = seq
                interleaved_seq[1::2] = site_seq+1
                formatted = ",".join(map(str,interleaved_seq))
                destination.write(formatted)
                destination.write('\n')

    bucket_op = b.bucket_batch_pad("formatted.txt", lower=200, interval=1000) # Large interval
    final_hidden, x_placeholder, batch_size_placeholder, seq_length_placeholder, initial_state_placeholder = (
    b.get_rep_ops())
    optimizer = tf.train.AdamOptimizer(0.001)
    if args.site_loss: 
        all_step_op = optimizer.minimize(b._loss+b.site_loss)
    else:
        print("No site loss!")
        all_step_op = optimizer.minimize(b._loss)


    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    losses, site_losses = [], [] 
    print(f"START OF TRAINING FOR {args.num_iters} ITERATIONS")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(args.num_iters):
            batch = sess.run(bucket_op)
            seq_batch = batch[:, ::2]
            site_seq_batch = batch[:, 1::2]-1
            if np.max(site_seq_batch) >= 3:
                print("WHOA.... SOMETHING WENT WRONG WITH THIS AMINO ACID:")
                print(seq_batch[np.argmax(np.max(site_seq_batch, axis=1))])
            length = nonpad_len(batch)-1
            if args.site_loss:
                loss, site_loss, _= sess.run([b._loss, b.site_loss, all_step_op],
                                               feed_dict = {
                                                x_placeholder: seq_batch[:, :-1],
                                                b._minibatch_y_placeholder: seq_batch[:, 1:],
                                                b._minibatch_site_y_placeholder: site_seq_batch[:, :-1],
                                                initial_state_placeholder: b._zero_state,
                                                seq_length_placeholder: length,
                                                batch_size_placeholder: batch_size,
                                                b._temp_placeholder: 0.1
                                               })
                losses.append(loss)
                site_losses.append(site_loss)
                print(f'Iteration {i}, loss {loss}, site loss {site_loss}')
            else:
                loss,_= sess.run([b._loss, all_step_op],
                                               feed_dict = {
                                                x_placeholder: seq_batch[:, :-1],
                                                b._minibatch_y_placeholder: seq_batch[:, 1:],
                                                initial_state_placeholder: b._zero_state,
                                                seq_length_placeholder: length,
                                                batch_size_placeholder: batch_size,
                                                b._temp_placeholder: 0.1
                                               })
                losses.append(loss)
                print(f'Iteration {i}, loss {loss}')
        b.dump_weights(sess, args.save_dir)

    fig, ax = plt.subplots(1,2, figsize=(13, 5))
    ax[0].plot(np.arange(args.num_iters), losses)
    ax[0].set_xlabel('iteration number')
    ax[0].set_ylabel('train loss')
    ax[0].set_title('Next amino acid prediction loss')

    if args.site_loss:
        ax[1].plot(np.arange(args.num_iters), site_losses)
        ax[1].set_xlabel('iteration number')
        ax[1].set_ylabel('train site loss')
        ax[1].set_title('Site type (normal/active/binding) loss')

    fig.savefig(f'{args.save_dir}/plot.png')
