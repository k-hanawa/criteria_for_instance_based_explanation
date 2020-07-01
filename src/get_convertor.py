#!/usr/bin/env python3 -u

import numpy as np
import chainer
from chainer import cuda

def mnist_convert(batch, device):
    x, y, *_ = zip(*batch)
    x = np.stack(x)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
    x = chainer.dataset.to_device(device, x)
    y = chainer.dataset.to_device(device, np.stack(y))
    return x, y

def cifar10_convert(batch, device):
    x, y, *_ = zip(*batch)
    x = np.stack(x)
    x = chainer.dataset.to_device(device, x)
    y = chainer.dataset.to_device(device, np.stack(y))
    return x, y

def text_convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif chainer.backends.cuda.get_device(device).id < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x)
                                     for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    xs, y, *_ = zip(*batch)
    xs = to_device_batch(xs)
    y = chainer.dataset.to_device(device, np.stack(y))
    return xs, y


def logreg_convert(batch, device):
    x, y, *_ = zip(*batch)
    x = np.stack(x)
    x = np.reshape(x, (x.shape[0], -1))
    x = chainer.dataset.to_device(device, x)
    y = chainer.dataset.to_device(device, np.stack(y))
    return x, y

def get_convertor(model, dataset):
    if model == 'logreg':
        return logreg_convert
    else:
        if dataset == 'mnist':
            return mnist_convert
        elif dataset == 'cifar10':
            return cifar10_convert
        else:
            return text_convert
