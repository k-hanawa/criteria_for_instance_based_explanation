#!/usr/bin/env python3 -u

import os
import json
from cnn import CNN
from lstm import LSTM
from logistic_regression import LogisticRegression

def get_mnist(model_name, setting, saved_dir, num_class=10):
    if model_name == 'cnn':
        out_channels = [int(n) for n in setting['out_dims'].split(' ')]
        ksizes = [int(n) for n in setting['filter_sizes'].split(' ')]
        pool_sizes = [int(n) for n in setting['pool_sizes'].split(' ')]
        model = CNN(
            input_channel=1,
            out_channels=out_channels,
            ksizes=ksizes,
            pool_sizes=pool_sizes,
            num_classes=num_class,
            out_dir=saved_dir
        )

    elif model_name == 'logreg':
        model = LogisticRegression(
            num_classes=num_class,
            out_dir=saved_dir
        )

    return model


def get_cifar10(model_name, setting, saved_dir, num_class=10):
    if model_name == 'cnn':
        out_channels = [int(n) for n in setting['out_dims'].split(' ')]
        ksizes = [int(n) for n in setting['filter_sizes'].split(' ')]
        pool_sizes = [int(n) for n in setting['pool_sizes'].split(' ')]
        model = CNN(
            input_channel=3,
            out_channels=out_channels,
            ksizes=ksizes,
            pool_sizes=pool_sizes,
            num_classes=num_class,
            out_dir=saved_dir
        )

    elif model_name == 'logreg':
        model = LogisticRegression(
            num_classes=num_class,
            out_dir=saved_dir
        )

    return model


def get_trec(model_name, setting, saved_dir, num_class=6):
    if model_name == 'lstm':
        with open(os.path.join(saved_dir, 'vocab.json')) as fi:
            vocab = json.load(fi)

        model = LSTM(
            n_vocab=len(vocab),
            n_emb=setting['n_emb'],
            n_layers=setting['n_layers'],
            n_dim=setting['n_dim'],
            num_classes=num_class,
            dropout=0,
            out_dir=saved_dir
        )

    elif model_name == 'logreg':
        model = LogisticRegression(
            num_classes=num_class,
            out_dir=saved_dir
        )

    return model


def get_cf(model_name, setting, saved_dir, num_class=6):
    if model_name == 'lstm':
        with open(os.path.join(saved_dir, 'vocab.json')) as fi:
            vocab = json.load(fi)

        model = LSTM(
            n_vocab=len(vocab),
            n_emb=setting['n_emb'],
            n_layers=setting['n_layers'],
            n_dim=setting['n_dim'],
            num_classes=num_class,
            dropout=0,
            out_dir=saved_dir
        )

    elif model_name == 'logreg':
        model = LogisticRegression(
            num_classes=num_class,
            out_dir=saved_dir
        )

    return model


get_model_fns = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'trec': get_trec,
    'cf': get_cf
}
