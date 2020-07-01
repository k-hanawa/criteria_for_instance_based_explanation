#!/usr/bin/env python3 -u

import chainer
import chainer.links as L
import chainer.functions as F
from general_model import GeneralModel

class LogisticRegression(GeneralModel):
    def __init__(self, num_classes, **kwargs):
        super(LogisticRegression, self).__init__(**kwargs)
        self.num_class = num_classes
        with self.init_scope():
            self.linear = L.Linear(None, num_classes)

    def __call__(self, x, y, enable_double_backprop=False):

        logit = self.predict(x, softmax=False, argmax=False)
        loss = F.softmax_cross_entropy(logit, y, enable_double_backprop=enable_double_backprop)
        accuracy = F.accuracy(logit, y)

        chainer.report({'loss': loss, 'acc': accuracy}, observer=self)

        return loss

    def predict(self, x, softmax=True, argmax=False):
        logit = self.linear(x)

        if argmax:
            return F.argmax(logit, axis=1)
        elif softmax:
            return F.softmax(logit)
        else:
            return logit

    def get_x(self, x, y):
        return chainer.cuda.to_cpu(x)

