#!/usr/bin/env python3 -u

import argparse
from tqdm import tqdm
import pickle

import metrics
import get_model
import get_convertor
import os
import json
import chainer
from chainer import serializers

import random
import numpy as np

import general_model
general_model.is_print = False

met_names_all = metrics.met_names
met_funcs_all = metrics.met_funcs
get_model_fns = get_model.get_model_fns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--saved-dir', '-s', type=str)
    parser.add_argument('--test-size', '-t', type=int, default=None)
    parser.add_argument('--metrics', '-i', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default=None)

    args = parser.parse_args()

    if args.model == 'lstm':
        chainer.config.use_cudnn = 'never'

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        chainer.cuda.cupy.random.seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)

    saved_dir = args.saved_dir
    gpu = args.gpu

    with open(os.path.join(saved_dir, 'encoded_data.pkl'), 'rb') as fi:
        all_data = pickle.load(fi)
    train, dev, test = all_data['train'], all_data['dev'], all_data['test']

    with open(os.path.join(saved_dir, 'setting.json')) as fi:
        setting = json.load(fi)

    model_file = os.path.join(saved_dir, 'best_model.npz')

    get_model_fn = get_model_fns[args.dataset]
    model = get_model_fn(args.model, setting, saved_dir)

    if gpu > -1:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    serializers.load_npz(model_file, model)

    convert = get_convertor.get_convertor(args.model, args.dataset)

    tmp_dict = {n: f for n, f in zip(met_names_all, met_funcs_all)}
    if args.metrics is None:
        met_names = met_names_all
    else:
        met_names = [n.strip() for n in args.metrics.split(',')]
    if args.model == 'logreg':
        excp = ['top_h_dot', 'top_h_cos', 'top_h_l2', 'all_h_dot', 'all_h_cos', 'all_h_l2']
        met_names = [n for n in met_names if n not in excp]
    met_funcs = [tmp_dict[n] for n in met_names]

    preds = []
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        test_iter = chainer.iterators.SerialIterator(train, batch_size=128, repeat=False, shuffle=False)
        for batch in test_iter:
            x, y = convert(batch, model.device)
            preds.append(chainer.cuda.to_cpu(model.predict(x, argmax=True).data))
    preds = np.hstack(preds)

    test_size = len(test) if args.test_size is None else args.test_size
    test_idxs = np.random.permutation(np.arange(len(train)))[:test_size]

    train_labels = [int(d[1]) for d in train]

    # N_target = 0
    # results = {n: 0 for n in met_names}
    all_results = []
    for test_idx in tqdm(test_idxs):
        target_data = train[test_idx]
        if preds[test_idx] != target_data[1]:
            continue
        # N_target += 1
        for met_func, met_name in zip(met_funcs, met_names):
            scores = met_func(target_data, train, model, convert)
            all_results.append({'idx': int(test_idx), 'y': int(target_data[1]),
                                'met': met_name, 'scores': np.array(scores)})

    if args.out is not None:
        with open(args.out, 'wb') as fo:
            tmp = {'results': all_results, 'train_labels': train_labels}
            pickle.dump(tmp, fo)


if __name__ == '__main__':
    main()
