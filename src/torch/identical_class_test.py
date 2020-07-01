#!/usr/bin/env python3 -u

import argparse
import json
import datetime
from chainer import cuda
from collections import Counter

import spacy
from tqdm import tqdm
import pickle
from random import shuffle

import os
import json

import torch

import random
import numpy as np

import src.torch_inf.mobilenetv2 as mobilenetv2

# general_model.is_print = False
def cossim(a, b):
    # a = a.astype(np.float64)
    # b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0

def l2sim(a, b):
    # a = a.astype(np.float64)
    # b = b.astype(np.float64)
    return -np.linalg.norm(a - b)

def grad_dot(target_data, train, model):
    predicted_loss_diffs = model.get_influence_on_test_loss(target_data,
                                                            train,
                                                            force_refresh=False,
                                                            matrix='none')

    return predicted_loss_diffs

def grad_cos(target_data, train, model):
    predicted_loss_diffs = model.get_influence_on_test_loss(target_data,
                                                            train,
                                                            force_refresh=False,
                                                            matrix='none',
                                                            sim_func=cossim)
    return predicted_loss_diffs

def grad_l2(target_data, train, model):
    predicted_loss_diffs = model.get_influence_on_test_loss(target_data,
                                                            train,
                                                            force_refresh=False,
                                                            matrix='none',
                                                            sim_func=l2sim)
    return predicted_loss_diffs

def multiclass_normalized(target_data, train, model):
    predicted_loss_diffs = model.get_influence_on_test_output(target_data,
                                                              train,
                                                              damping=1e-10,
                                                              force_refresh=False)
    return predicted_loss_diffs

def x_sim_dot(target_data, train, model):
    predicted_loss_diffs = model.get_influence_by_hsim(target_data,
                                                       train,
                                                       'x',
                                                       sim_func=np.dot)
    return predicted_loss_diffs

def x_sim_cos(target_data, train, model):
    predicted_loss_diffs = model.get_influence_by_hsim(target_data,
                                                       train,
                                                       'x',
                                                       sim_func=cossim)
    return predicted_loss_diffs

def x_sim_l2(target_data, train, model):
    predicted_loss_diffs = model.get_influence_by_hsim(target_data,
                                                       train,
                                                       'x',
                                                       sim_func=l2sim)
    return predicted_loss_diffs

def top_h_dot(target_data, train, model):
    predicted_loss_diffs = model.get_influence_by_hsim(target_data,
                                                       train,
                                                       'top_h',
                                                       sim_func=np.dot)
    return predicted_loss_diffs

def top_h_cos(target_data, train, model):
    predicted_loss_diffs = model.get_influence_by_hsim(target_data,
                                                       train,
                                                       'top_h',
                                                       sim_func=cossim)
    return predicted_loss_diffs

def top_h_l2(target_data, train, model):
    predicted_loss_diffs = model.get_influence_by_hsim(target_data,
                                                       train,
                                                       'top_h',
                                                       sim_func=l2sim)
    return predicted_loss_diffs

def all_h_dot(target_data, train, model):
    predicted_loss_diffs = model.get_influence_by_hsim(target_data,
                                                       train,
                                                       'all_h',
                                                       sim_func=np.dot)
    return predicted_loss_diffs

def all_h_cos(target_data, train, model):
    predicted_loss_diffs = model.get_influence_by_hsim(target_data,
                                                       train,
                                                       'all_h',
                                                       sim_func=cossim)
    return predicted_loss_diffs

def all_h_l2(target_data, train, model):
    predicted_loss_diffs = model.get_influence_by_hsim(target_data,
                                                       train,
                                                       'all_h',
                                                       sim_func=l2sim)
    return predicted_loss_diffs

inf_funcs = [grad_dot, grad_cos, grad_l2, x_sim_dot, x_sim_cos, x_sim_l2, top_h_dot, top_h_cos, top_h_l2, all_h_dot, all_h_cos, all_h_l2]
inf_names = ['grad_dot', 'grad_cos', 'grad_l2', 'x_sim_dot', 'x_sim_cos', 'x_sim_l2', 'top_h_dot', 'top_h_cos', 'top_h_l2', 'all_h_dot', 'all_h_cos', 'all_h_l2']


def main():
    parser = argparse.ArgumentParser(description='Attention-based NMT')
    parser.add_argument('--saved-dir', '-s', type=str,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--test-size', '-t', type=int, default=None,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--map-at-k', '-k', type=int, default=10,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--seed', type=int, default=0,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--out-all', default=None,
                        help='directory to output the result')

    args = parser.parse_args()


    random.seed(args.seed)
    np.random.seed(args.seed)

    saved_dir = args.saved_dir
    gpu = args.gpu


    data_params = {'path': 'data/cifar10_torchvision', 'batch_size': 1}

    train_loader, validation_loader = mobilenetv2.make_dataloaders(data_params)

    data_params = {'path': 'data/cifar10_torchvision', 'batch_size': 32}
    train_loader_batch, validation_loader_batch = mobilenetv2.make_dataloaders(data_params)

    data_params = {'path': 'data/cifar10_torchvision', 'batch_size': 1}
    train_loader_raw, validation_loader_raw = mobilenetv2.make_dataloaders(data_params, raw=True)

    train_params = {'description': 'Test',
                    'num_epochs': 3,
                    'check_point': 50, 'device': 'cuda',
                    'train_loader': train_loader, 'validation_loader': validation_loader}

    # res = test_model(model, train_params)
    # print(res)

    with open(os.path.join(args.saved_dir, 'train_idxs.pkl'), 'rb') as fi:
        train_idxs = pickle.load(fi)

    test = [d for d in validation_loader]
    train = [[d for d in train_loader][i] for i in train_idxs]
    test_raw = [d for d in validation_loader_raw]
    train_raw = [[d for d in train_loader_raw][i] for i in train_idxs]

    model = mobilenetv2.mobilenet_v2(pretrained=True, model_path=os.path.join(args.saved_dir, 'best_model.pt'))
    model.out_dir = saved_dir

    if gpu > -1:
        device = 'cuda:{}'.format(gpu)
    else:
        device = 'cpu'
    model = model.to(device).eval()
    model.device = device

    preds = []
    with torch.no_grad():
        for image, label in tqdm(validation_loader_batch):
            image = image.to(model.device)
            prediction = model(image)
            preds.append(np.argmax(prediction.cpu().numpy(), axis=1))
    preds = np.hstack(preds)

    test_size = len(test) if args.test_size is None else args.test_size
    test_idxs = np.random.permutation(np.arange(len(test)))[:test_size]

    train_labels = [int(d[1][0]) for d in train]

    # N_target = 0
    # results = {n: 0 for n in inf_names}
    all_results = []
    for test_idx in tqdm(test_idxs):
        target_data = test[test_idx]
        target_data_raw = test_raw[test_idx]
        # N_target += 1
        for inf_func, inf_name in zip(inf_funcs, inf_names):
            if inf_name.startswith('x_sim'):
                infs = inf_func(target_data_raw, train_raw, model)
            else:
                infs = inf_func(target_data, train, model)

            all_results.append({'idx': int(test_idx), 'y': int(target_data[1][0]),
                                'met': inf_name, 'scores': np.array(infs)})

    if args.out is not None:
        with open(args.out, 'wb') as fo:
            tmp = {'results': all_results, 'train_labels': train_labels}
            pickle.dump(tmp, fo)


if __name__ == '__main__':
    main()
