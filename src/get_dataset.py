#!/usr/bin/env python3 -u

import os
import json
import numpy as np
from random import shuffle, sample
from collections import Counter
from tqdm import tqdm

def get_mnist_data(train_size=1., merge=False):
    from sklearn.datasets import fetch_openml
    data_sets = fetch_openml('mnist_784')
    x = np.reshape((data_sets.data.astype(np.float32) / 255), (data_sets.data.shape[0], 28, 28))

    if merge:
        label_vals = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        shuffle(label_vals)
        label_dic = {k: v for k, v in zip(range(10), label_vals)}
        y = np.array([label_dic[int(t)] for t in data_sets.target], dtype=np.int32)
        orig_y = np.array([int(t) for t in data_sets.target], dtype=np.int32)
        train = [(_x, _y, _oy) for _x, _y, _oy in zip(x[:55000], y[:55000], orig_y[:55000])]
        dev = [(_x, _y, _oy) for _x, _y, _oy in zip(x[55000:60000], y[55000:60000], orig_y[55000:60000])]
        test = [(_x, _y, _oy) for _x, _y, _oy in zip(x[60000:], y[60000:], orig_y[60000:])]
    else:
        y = np.array([int(t) for t in data_sets.target], dtype=np.int32)
        label_dic = {i: i for i in range(10)}
        train = [(_x, _y) for _x, _y in zip(x[:55000], y[:55000])]
        dev = [(_x, _y) for _x, _y in zip(x[55000:60000], y[55000:60000])]
        test = [(_x, _y) for _x, _y in zip(x[60000:], y[60000:])]
    train = sample(train, int(len(train) * train_size))

    return train, dev, test, label_dic


def get_cifar10_data(train_size=1., merge=False):
    # from keras.datasets import cifar10
    from sklearn.datasets import fetch_openml
    data_sets = fetch_openml('CIFAR_10')
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
    Y_train = np.reshape(Y_train, (Y_train.shape[0],)).astype(np.int32)

    X_test = X_test.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
    Y_test = np.reshape(Y_test, (Y_test.shape[0],)).astype(np.int32)


    if merge:
        label_vals = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        shuffle(label_vals)
        label_dic = {k: v for k, v in zip(range(10), label_vals)}
        train = [(x, label_dic[int(y)], class_names[int(y)]) for x, y in zip(X_train, Y_train)]
        dev = [(x, label_dic[int(y)], class_names[int(y)]) for x, y in zip(X_test, Y_test)]
        test = [(x, label_dic[int(y)], class_names[int(y)]) for x, y in zip(X_test, Y_test)]
    else:
        label_dic = {n: i for i, n in enumerate(class_names)}
        train = [(x, y) for x, y in zip(X_train, Y_train)]
        dev = [(x, y) for x, y in zip(X_test, Y_test)]
        test = [(x, y) for x, y in zip(X_test, Y_test)]
    train = sample(train, int(len(train) * train_size))
    return train, dev, test, label_dic


def get_trec_data(train_size=1., merge=False, model='', path='', threshold=0):
    import spacy
    nlp = spacy.load('en_core_web_sm')
    BOS, EOS, UNK = 0, 1, 2

    def make_vocab(data, thresholod):
        c = Counter()
        for tokens in data:
            c.update(tokens)
        vocab = {'<bos>': BOS, '<eos>': EOS, '<unk>': UNK}
        idx = 3
        for k, v in c.most_common():
            if v < thresholod:
                break
            vocab[k] = idx
            idx += 1
        return vocab

    label_dic = {}

    def read_data(_path):
        sentences = []
        labels = []
        with open(_path) as f:
            for line in f:
                label, words = line.rstrip('\n').split(' ', 1)
                label = label.split(':')[0]
                if label not in label_dic:
                    label_dic[label] = len(label_dic)
                labels.append(int(label_dic[label]))
                sentences.append(words)
        return sentences, labels

    train_sentences, train_labels = read_data(os.path.join(path, 'train.txt'))
    test_sentences, test_labels = read_data(os.path.join(path, 'test.txt'))

    def tokenize(data):
        tokens_list = []
        for txt in tqdm(data):
            doc = nlp(txt, disable=['parser', 'tagger', 'ner'])
            tokens = [t.lower_ for t in doc]
            # tokens = list(txt)
            tokens_list.append(tokens)
        return tokens_list

    train_tokens = tokenize(train_sentences)
    test_tokens = tokenize(test_sentences)

    vocab = make_vocab(train_tokens, threshold)

    if merge:
        # class_names = ["LOC", "NUM", "ABBR", "DESC", "HUM", "ENTY"]
        label_vals = [0, 0, 0, 1, 1, 1]
        shuffle(label_vals)
        label_dic = {k: v for k, v in zip(label_dic, label_vals)}
        if model == 'logreg':
            def make_dataset(sentences, labels):
                dataset = []
                assert len(sentences) == len(labels)
                for sentence, y in zip(sentences, labels):
                    vec = np.zeros(len(vocab), dtype=np.float32)
                    for t in sentence:
                        if t in vocab:
                            vec[vocab[t]] += 1
                    dataset.append((vec, label_dic[y], y))
                return dataset
        else:
            def make_dataset(sentences, labels):
                dataset = []
                assert len(sentences) == len(labels)
                for sentence, y in zip(sentences, labels):
                    x = [BOS] + [vocab.get(t, UNK) for t in sentence] + [EOS]
                    x = np.array(x, dtype=np.int32)
                    dataset.append((x, label_dic[y], y))
                return dataset

    else:
        if model == 'logreg':
            def make_dataset(sentences, labels):
                dataset = []
                assert len(sentences) == len(labels)
                for sentence, y in zip(sentences, labels):
                    vec = np.zeros(len(vocab), dtype=np.float32)
                    for t in sentence:
                        if t in vocab:
                            vec[vocab[t]] += 1
                    dataset.append((vec, y))
                return dataset
        else:
            def make_dataset(sentences, labels):
                dataset = []
                assert len(sentences) == len(labels)
                for sentence, y in zip(sentences, labels):
                    x = [BOS] + [vocab.get(t, UNK) for t in sentence] + [EOS]
                    x = np.array(x, dtype=np.int32)
                    dataset.append((x, y))
                return dataset

    train = make_dataset(train_tokens, train_labels)
    dev = make_dataset(test_tokens, test_labels)
    test = make_dataset(test_tokens, test_labels)
    train = sample(train, int(len(train) * train_size))
    max_len = max([d[0].shape[0] for d in train])
    padded_train_x = [np.pad(np.sort(d[0]), (0, max_len - d[0].shape[0]), mode='constant', constant_values=-1) for d in train]
    train_x, u_idxs, inv_idxs = np.unique(padded_train_x, axis=0, return_index=True, return_inverse=True)
    uniq_train = []
    done_idx = set()
    for d, idx in zip(train, inv_idxs):
        if idx in done_idx:
            continue
        uniq_train.append(train[u_idxs[idx]])
        done_idx.add(idx)
    train = uniq_train

    return train, dev, test, label_dic, vocab


def get_cf_data(train_size=1., merge=False, model='', path='data/cf', threshold=0):
    import spacy
    nlps = {}
    nlps['en'] = spacy.load('en_core_web_sm')
    nlps['es'] = spacy.load('es_core_news_sm')
    nlps['fr'] = spacy.load('fr_core_news_sm')
    nlps['jp'] = spacy.load('ja_ginza')
    BOS, EOS, UNK = 0, 1, 2

    def make_vocab(data, thresholod):
        c = Counter()
        for tokens in data:
            c.update(tokens)
        vocab = {'<bos>': BOS, '<eos>': EOS, '<unk>': UNK}
        idx = 3
        for k, v in c.most_common():
            if v < thresholod:
                break
            vocab[k] = idx
            idx += 1
        return vocab

    label_dic = {}
    def read_data(_path):
        sentences = []
        labels = []
        with open(_path) as f:
            for line in f:
                if line.strip() == '':
                    continue
                _, text, label = line.rstrip('\n').split('\t')
                if ',' in label:
                    continue
                if label not in label_dic:
                    label_dic[label] = len(label_dic)
                labels.append(int(label_dic[label]))
                sentences.append(text)
        return sentences, labels

    def tokenize(data, lang):
        tokens_list = []
        for txt in tqdm(data):
            doc = nlps[lang](txt, disable=['parser', 'tagger', 'ner'])
            tokens = [t.lower_ for t in doc]
            # tokens = list(txt)
            tokens_list.append(tokens)
        return tokens_list

    def read_and_tokenize(data_name):
        tokens, labels = [], []
        for lang in ('en', 'es', 'fr', 'jp'):
            _sentences, _labels = read_data(os.path.join(path, '{}-{}.txt'.format(lang, data_name)))

            _tokens = tokenize(_sentences, lang)

            tokens.extend(_tokens)
            labels.extend(_labels)
        return tokens, labels

    train_tokens, train_labels = read_and_tokenize('training')
    dev_tokens, dev_labels = read_and_tokenize('development')
    test_tokens, test_labels = read_and_tokenize('test-oracle')

    vocab = make_vocab(train_tokens, threshold)

    if merge:
        # class_names = ["complaint", "meaningless", "comment", "request", "undetermined", "bug"]
        label_vals = [0, 0, 0, 1, 1, 1]
        shuffle(label_vals)
        label_dic = {k: v for k, v in zip(label_dic, label_vals)}
        if model == 'logeg':
            def make_dataset(sentences, labels):
                dataset = []
                assert len(sentences) == len(labels)
                for sentence, y in zip(sentences, labels):
                    vec = np.zeros(len(vocab), dtype=np.float32)
                    for t in sentence:
                        if t in vocab:
                            vec[vocab[t]] += 1
                    dataset.append((vec, label_dic[y], y))
                return dataset
        else:
            def make_dataset(sentences, labels):
                dataset = []
                assert len(sentences) == len(labels)
                for sentence, y in zip(sentences, labels):
                    x = [BOS] + [vocab.get(t, UNK) for t in sentence] + [EOS]
                    x = np.array(x, dtype=np.int32)
                    dataset.append((x, label_dic[y], y))
                return dataset

    else:
        if model == 'logreg':
            def make_dataset(sentences, labels):
                dataset = []
                assert len(sentences) == len(labels)
                for sentence, y in zip(sentences, labels):
                    vec = np.zeros(len(vocab), dtype=np.float32)
                    for t in sentence:
                        if t in vocab:
                            vec[vocab[t]] += 1
                    dataset.append((vec, y))
                return dataset
        else:
            def make_dataset(sentences, labels):
                dataset = []
                assert len(sentences) == len(labels)
                for sentence, y in zip(sentences, labels):
                    x = [BOS] + [vocab.get(t, UNK) for t in sentence] + [EOS]
                    x = np.array(x, dtype=np.int32)
                    dataset.append((x, y))
                return dataset

    train = make_dataset(train_tokens, train_labels)
    dev = make_dataset(dev_tokens, dev_labels)
    test = make_dataset(test_tokens, test_labels)
    train = sample(train, int(len(train) * train_size))
    max_len = max([d[0].shape[0] for d in train])
    padded_train_x = [np.pad(np.sort(d[0]), (0, max_len - d[0].shape[0]), mode='constant', constant_values=-1) for d in train]
    train_x, u_idxs, inv_idxs = np.unique(padded_train_x, axis=0, return_index=True, return_inverse=True)
    uniq_train = []
    done_idx = set()
    for d, idx in zip(train, inv_idxs):
        if idx in done_idx:
            continue
        uniq_train.append(train[u_idxs[idx]])
        done_idx.add(idx)
    train = uniq_train

    return train, dev, test, label_dic, vocab


def get_dataset(dataset, train_size=1., merge=False, model='', path='', threshold=0):
    if dataset == 'mnist':
        train, dev, test, label_dic = get_mnist_data(train_size, merge)
        return train, dev, test, label_dic, None
    elif dataset == 'cifar10':
        train, dev, test, label_dic = get_cifar10_data(train_size, merge)
        return train, dev, test, label_dic, None
    elif dataset == 'trec':
        return get_trec_data(train_size, merge, model, path, threshold)
    elif dataset == 'cf':
        return get_cf_data(train_size, merge, model, path, threshold)

