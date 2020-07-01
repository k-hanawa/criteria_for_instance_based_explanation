#!/usr/bin/env python3 -u

import chainer
import chainer.links as L
import chainer.functions as F
from general_model import GeneralModel
import numpy as np

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class LSTM(GeneralModel):
    def __init__(self, n_vocab, n_emb, n_layers, n_dim, num_classes, dropout=0.5, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.n_vocab = n_vocab
        self.num_class = num_classes
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_emb)
            self.bilstm = L.NStepBiLSTM(n_layers, n_emb, n_dim, dropout)
            self.l1 = L.Linear(n_dim * 2, num_classes)
            # self.l1 = L.Linear(n_dim, num_classes)

    def __call__(self, xs, y, enable_double_backprop=False):

        logit = self.predict(xs, softmax=False, argmax=False)
        loss = F.softmax_cross_entropy(logit, y, enable_double_backprop=enable_double_backprop)
        accuracy = F.accuracy(logit, y)

        chainer.report({'loss': loss, 'acc': accuracy}, observer=self)

        return loss

    def predict(self, xs, softmax=True, argmax=False):
        exs = sequence_embed(self.embed, xs)
        hy, cy, ys = self.bilstm(None, None, exs)
        h = F.concat(hy[-2:])
        logit = self.l1(h)
        # logit = self.l1(h)

        if argmax:
            return F.argmax(logit, axis=1)
        elif softmax:
            return F.softmax(logit)
        else:
            return logit

    def get_x(self, xs, y):
        zeros = self.xp.zeros((len(xs), self.n_vocab), dtype=self.xp.int32)
        for i, x in enumerate(xs):
            for _x in x:
                zeros[i, _x] += 1
        return chainer.cuda.to_cpu(zeros)

    def get_top_h(self, xs, y):
        exs = sequence_embed(self.embed, xs)
        hy, cy, ys = self.bilstm(None, None, exs)
        h = F.concat(hy[-2:])
        return chainer.cuda.to_cpu(h.data)

    def get_all_h(self, xs, y):
        exs = sequence_embed(self.embed, xs)
        hy, cy, ys = self.bilstm(None, None, exs)
        h = F.concat(hy)
        return chainer.cuda.to_cpu(h.data)
