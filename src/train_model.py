#!/usr/bin/env python3 -u

import argparse
import os
import json
import datetime
import random

import numpy as np
import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

import get_dataset
import get_model
import get_convertor

get_model_fns = get_model.get_model_fns

def main():
    current_datetime = '{}'.format(datetime.datetime.today())

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--model', default='cnn')
    parser.add_argument('--merge_label', action='store_true')
    parser.add_argument('--train_size', type=float, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batchsize', '-b', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--pretrained_model', default=None)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='result')
    # for cnn
    parser.add_argument('--out_dims', type=str, default='16 16 16')
    parser.add_argument('--filter_sizes', type=str, default='3 3 3')
    parser.add_argument('--pool_sizes', type=str, default='2 2 2')
    parser.add_argument('--dropout', type=float, default=0.1)
    # for lstm
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_emb', type=int, default=8)
    parser.add_argument('--n_dim', type=int, default=16)
    # for nlp
    parser.add_argument('--min_vocab', type=int, default=5)

    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    random.seed(args.seed)
    np.random.seed(args.seed)
    chainer.cuda.cupy.random.seed(args.seed)

    train, dev, test, label_dic, vocab = get_dataset.get_dataset(args.dataset, args.train_size, merge=args.merge_label, model=args.model, path=args.dataset_path, threshold=args.min_vocab)
    convert = get_convertor.get_convertor(args.model, args.dataset)

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    current = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current, args.out, 'best_model.npz')
    model_setup = args.__dict__.copy()
    model_setup['model_path'] = model_path
    model_setup['datetime'] = current_datetime

    with open(os.path.join(args.out, 'setting.json'), 'w') as f:
        json.dump(model_setup, f, ensure_ascii=False)

    if vocab is not None:
        model_setup['vocab_path'] = os.path.join(args.out, 'vocab.json')
        with open(os.path.join(args.out, 'vocab.json'), 'w') as f:
            json.dump(vocab, f, ensure_ascii=False)

    model_setup['label_path'] = os.path.join(args.out, 'label.json')
    with open(os.path.join(args.out, 'label.json'), 'w') as f:
        json.dump(label_dic, f, ensure_ascii=False)

    model_setup['data_path'] = os.path.join(args.out, 'encoded_data.pkl')
    import pickle
    all_data = {'train': train, 'dev': dev, 'test': test}
    with open(model_setup['data_path'], mode='wb') as f:
        pickle.dump(all_data, f)

    get_model_fn = get_model_fns[args.dataset]
    setting = args.__dict__.copy()
    if args.merge_label:
        model = get_model_fn(args.model, setting, args.out, num_class=2)
    else:
        model = get_model_fn(args.model, setting, args.out)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    if args.pretrained_model is not None:
        serializers.load_npz(args.pretrained_model, model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.005))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert,
        device=args.gpu
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(
        extensions.LogReport(trigger=(1, 'epoch'))
    )
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/acc', 'validation/main/acc', 'elapsed_time']
        ),
        trigger=(1, 'epoch')
    )
    trainer.extend(
        extensions.snapshot_object(model, filename='model_epoch_{.updater.epoch}.npz'),
        trigger=(10, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    eval_model = model.copy()
    val_iter = chainer.iterators.SerialIterator(dev, args.batchsize, repeat=False, shuffle=False)
    trainer.extend(extensions.Evaluator(val_iter, eval_model,
                                        converter=convert,
                                        device=args.gpu))
    record_trigger = training.triggers.MaxValueTrigger('validation/main/acc', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'best_model.npz'), trigger=record_trigger)


    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    current = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current, args.out, 'best_model.npz')
    model_setup = args.__dict__.copy()
    model_setup['model_path'] = model_path
    model_setup['datetime'] = current_datetime

    with open(os.path.join(args.out, 'setting.json'), 'w') as f:
        json.dump(model_setup, f, ensure_ascii=False)

    if vocab is None:
        model_setup['vocab_path'] = os.path.join(args.out, 'vocab.json')
        with open(os.path.join(args.out, 'vocab.json'), 'w') as f:
            json.dump(vocab, f, ensure_ascii=False)

    model_setup['label_path'] = os.path.join(args.out, 'label.json')
    with open(os.path.join(args.out, 'label.json'), 'w') as f:
        json.dump(label_dic, f, ensure_ascii=False)

    model_setup['data_path'] = os.path.join(args.out, 'encoded_data.pkl')
    import pickle
    all_data = {'train': train, 'dev': dev, 'test': test}
    with open(model_setup['data_path'], mode='wb') as f:
        pickle.dump(all_data, f)

    print('start training')
    trainer.run()

if __name__ == '__main__':
    main()
