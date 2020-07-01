#!/usr/bin/env python3 -u

import argparse
from tqdm import tqdm
import pickle
import os
import json
import numpy as np
from collections import defaultdict


met_names = [
             'x_sim_dot',
             'x_sim_cos',
             'x_sim_l2',
             'top_h_dot',
             'top_h_cos',
             'top_h_l2',
             'all_h_dot',
             'all_h_cos',
             'all_h_l2',
             'influence_dot',
             'influence_cos',
             'influence_l2',
             'fisher_dot',
             'fisher_cos',
             'fisher_l2',
             'grad_dot',
             'grad_cos',
             'grad_l2'
             ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--out', '-o', type=str)

    args = parser.parse_args()

    N_target = defaultdict(int)
    sums = defaultdict(int)
    with open(args.input, 'rb') as fi:
        all_results = pickle.load(fi)
    results = all_results['results']
    for res in tqdm(results, leave=False):
        scores = res['scores']
        top_idx = np.argsort(scores)[::-1][0]
        if res['idx'] == top_idx:
            sums[res['met']] += 1
        else:
            sums[res['met']] += 0
        N_target[res['met']] += 1
    with open(args.out, 'w') as fo:
        for k in sums:
            if k in met_names:
                fo.write('{}\t{}\n'.format(k, sums[k] / N_target[k]))

if __name__ == '__main__':
    main()
