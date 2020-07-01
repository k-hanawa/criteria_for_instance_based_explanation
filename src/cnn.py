#!/usr/bin/env python3 -u

import chainer
import chainer.links as L
import chainer.functions as F
from general_model import GeneralModel

class CNN(GeneralModel):
    def __init__(self, input_channel, out_channels, ksizes, pool_sizes, num_classes, dropout=0, **kwargs):
        super(CNN, self).__init__(**kwargs)
        assert len(out_channels) == len(ksizes) == len(pool_sizes)
        self.num_layers = len(out_channels)
        self.pool_sizes = pool_sizes
        self.dropout = dropout
        self.num_class = num_classes
        with self.init_scope():
            in_channels = [input_channel] + out_channels[:-1]
            for i, (in_channel, out_channel, ksize) in enumerate(zip(in_channels, out_channels, ksizes)):
                pad = int(ksize / 2)
                setattr(self, 'conv{}_1'.format(i), L.Convolution2D(in_channel, out_channel, ksize, pad=pad))
                setattr(self, 'conv{}_2'.format(i), L.Convolution2D(out_channel, out_channel, ksize, pad=pad))
            self.l1 = L.Linear(None, num_classes)

    def __call__(self, x, y, enable_double_backprop=False):

        logit = self.predict(x, softmax=False, argmax=False)
        loss = F.softmax_cross_entropy(logit, y, enable_double_backprop=enable_double_backprop)
        accuracy = F.accuracy(logit, y)

        chainer.report({'loss': loss, 'acc': accuracy}, observer=self)

        return loss

    def predict(self, x, softmax=True, argmax=False):
        h = x
        for i in range(self.num_layers):
            h = F.relu(getattr(self, 'conv{}_1'.format(i))(h))
            h = F.relu(getattr(self, 'conv{}_2'.format(i))(h))
            h = F.dropout(h, self.dropout)
            h = F.max_pooling_2d(h, self.pool_sizes[i])
        h = F.average(h, axis=(2, 3))

        logit = self.l1(h)

        if argmax:
            return F.argmax(logit, axis=1)
        elif softmax:
            return F.softmax(logit)
        else:
            return logit

    def get_x(self, x, y):
        x = F.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        return chainer.cuda.to_cpu(x.data)

    def get_top_h(self, x, y):
        h = x
        for i in range(self.num_layers):
            h = F.relu(getattr(self, 'conv{}_1'.format(i))(h))
            h = F.relu(getattr(self, 'conv{}_2'.format(i))(h))
            h = F.max_pooling_2d(h, self.pool_sizes[i])
        h = F.average(h, axis=(2, 3))
        return chainer.cuda.to_cpu(h.data)

    def get_all_h(self, x, y):
        hs = []
        h = x
        # hs.append(F.reshape(h, (h.shape[0], h.shape[1] * h.shape[2] * h.shape[3])))
        for i in range(self.num_layers):
            h = F.relu(getattr(self, 'conv{}_1'.format(i))(h))
            hs.append(F.reshape(h, (h.shape[0], h.shape[1] * h.shape[2] * h.shape[3])))
            h = F.relu(getattr(self, 'conv{}_2'.format(i))(h))
            hs.append(F.reshape(h, (h.shape[0], h.shape[1] * h.shape[2] * h.shape[3])))
            h = F.max_pooling_2d(h, self.pool_sizes[i])
            hs.append(F.reshape(h, (h.shape[0], h.shape[1] * h.shape[2] * h.shape[3])))
        h = F.average(h, axis=(2, 3))
        hs.append(h)
        ret_h = F.hstack(hs)
        return chainer.cuda.to_cpu(ret_h.data)
