import os
import sys
import pickle
import argparse
import pandas as pd
from sklearn.manifold import TSNE

from tsne_hack import extract_sequence
from visualize import savegif


def manual_extract(df,iters = 250, early = 0, perplexity = 500, random_state = 1):    
    # df = pd.read_csv(r'../ppm-on-time-new-data.csv')
    # df = df.loc[df['Primary Delay per 100 miles']<4]
    # df.pop('Date')
    #labels = df.pop('On Time WTT%')
    
    # df = df[['Primary Delay per 100 miles', 'Footfall','Count of Trains/Timing Points - WTT', 'Planned']]
    
    X = df.values
    #labels = labels.values
    
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    
    
    tsne = TSNE(n_iter=iters, verbose=True, n_jobs = -1, perplexity=  perplexity, random_state = random_state)
    tsne._EXPLORATION_N_ITER = early
    
    Y_seq = extract_sequence(tsne, X)
    
    return Y_seq

def main(args):
    #data_path = './data/%s.pkl' % args.dataset 
    #with open(data_path, 'rb') as f:
    #    X, labels = pickle.load(f)
    
    
    df = pd.read_csv(r'../ppm-on-time-new-data.csv')
    df = df.loc[df['Primary Delay per 100 miles']<4]
    df.pop('Date')
    labels = df.pop('On Time WTT%')
    
    df = df[['Primary Delay per 100 miles', 'Footfall','Count of Trains/Timing Points - WTT', 'Planned', 'On Time - WTT']]
    
    X = df.values
    labels = labels.values
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    
    tsne = TSNE(n_iter=args.num_iters, verbose=True, n_jobs = 1, perplexity=  500)
    tsne._EXPLORATION_N_ITER = args.early_iters
    
    Y_seq = extract_sequence(tsne, X)
    with open('results/res.pkl', 'wb') as f:
        pickle.dump(Y_seq, f)

    if not os.path.exists('figures'):
        os.mkdir('figures')

    lo = Y_seq.min(axis=0).min(axis=0).max()
    hi = Y_seq.max(axis=0).max(axis=0).min()
    lo = -25
    hi = 25
    
    limits = ([lo, hi], [lo, hi])
    fig_name = '%s-%d-%d-tsne' % (args.dataset, args.num_iters, args.early_iters)
    fig_path = './figures/%s.gif' % fig_name
    savegif(Y_seq, labels, fig_name, fig_path, limits=limits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ppm-on-time-new-data')
    parser.add_argument('--num_iters', type=int, default=250)
    parser.add_argument('--early_iters', type=int, default=0)
    args = parser.parse_args()
    main(args)
#%%

# import csv
# import pandas as pd

# df = pd.read_csv(r'../ppm-on-time-new-data.csv')
# df = df.loc[df['Primary Delay per 100 miles']<4]

# date = df.pop('Date')
# labels = df.pop('On Time WTT%')

# df = df[['Primary Delay per 100 miles', 'Footfall','Count of Trains/Timing Points - WTT', 'Planned']]

# vectors = df.values      # load your embeddings
# metadata = date.values  # load your metadata

# with open('../output.tsv', 'wt') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     for vector in vectors:
#       tsv_writer.writerow(vector)

# with open('../metadata.tsv', 'wt') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     for meta in metadata:
#       tsv_writer.writerow([meta])