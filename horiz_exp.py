import os
import json
import random
import uuid
import argparse
import numpy as np
from ts2vec import TS2Vec
import datautils
import torch
from sklearn.preprocessing import normalize

def set_seeds(seed):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed + 111)
    _ = torch.cuda.manual_seed(seed + 222)
    _ = random.seed(seed + 333)

# -
# Cli

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim',    type=int,   default=128)
    parser.add_argument('--q_prop',     type=float, default=.3)
    parser.add_argument('--q_samples',  type=int,   default=50)
    parser.add_argument('--n_samples',  type=int,   default=1_000)
    parser.add_argument('--n_epochs',   type=int,   default=50)
    parser.add_argument('--out_path',   type=str,   default='results/')
    parser.add_argument('--seed',       type=int,   default=123123)
    return parser.parse_args() 
args = parse_args()
set_seeds(args.seed)
if args.out_path is not None:
    os.makedirs(args.out_path, exist_ok=True)

# - 
# Data

pros = np.load(os.path.join('/home/ubuntu/projects/redem/data/horizons/dtm_swiss_10m_v2/', 'pros.npy'))
mean = np.nanmean(pros)
std  = np.nanstd(pros)
pros = (pros - mean) / std
pros = np.expand_dims(pros, -1)[0:args.n_samples]

if args.n_samples > pros.shape[0]:
    args.n_samples = pros.shape[0]

if args.q_samples == None:
    args.q_samples = args.n_samples


def sample_queries(x, q_prop=1., loop=False):
    if loop == False:
        q_dim = int(x.shape[1]*q_prop)
        if x.shape[1]-q_dim > 0:
            starts = np.random.choice(x.shape[1]-q_dim, size=x.shape[0], replace=True)
        else:
            starts = np.array([0]*x.shape[0])
        q = np.array([x[i, s:s+q_dim] for i,s in enumerate(starts)])
    return q

pros  = pros[0:args.n_samples]
q_idx = np.arange(args.n_samples)[np.random.choice(args.n_samples, size=args.q_samples)]
q     = sample_queries(pros[q_idx], q_prop=args.q_prop)


# -
# Train 

model = TS2Vec(
    input_dims  = 1,
    device      = 0,
    output_dims = args.emb_dim
)

loss_log = model.fit(
    pros,
    verbose  = True,
    n_epochs = args.n_epochs
)

# -
# Embed

print(pros.shape)

print("full series encoding")

len_window = int(pros.shape[1] * .5)
X = model.encode(pros, encoding_window=len_window)
n_reps = X.shape[1]
X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
# X = model.encode(pros, encoding_window='full_series')
Q = model.encode(q, encoding_window='full_series')

# -
# Compute dists  + Metric

X = normalize(X)
Q = normalize(Q)
dists = (Q @ X.T)

acc_k = {}
for k in [1,10,20,50]:
    topk = np.argpartition(-dists, k, axis=1)[:,:k]
    acc  = np.zeros(args.q_samples)
    for i in range(args.q_samples):
        topk_i = set(topk[i,:])
        true_p = set(np.arange(q_idx[i]*n_reps, q_idx[i]*n_reps+n_reps))
        print("topk_i", topk_i, "true_p", true_p)
        pos    = topk_i.intersection(true_p)
        if len(pos) > 0:
            acc[i] = 1
    acc_k[f'acc_{k}'] = acc.sum()/args.q_samples
print(acc_k)

# for k in [1, 3, 5]:
#     topk = np.argpartition(-dists, 1, axis=1)[:,:k]
#     acc  = np.zeros(args.q_samples)
#     for kk in range(k):
#         acc += (topk[:,kk] == (q_idx).astype(int))
#     acc_k[f'acc_{k}'] = acc.sum()/args.q_samples


# results = vars(args)
# results.update(acc_k)
# filename = str(uuid.uuid4())
# json.dump(results, open(os.path.join(args.out_path, f'{filename}.json'), "w"))


# acc_k = {}

