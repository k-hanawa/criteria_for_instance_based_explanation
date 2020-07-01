import torch
import torch.nn as nn
import os

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm as pbar
import numpy as np
import time

from tqdm import tqdm

import scipy.linalg

import random
from random import shuffle

is_print = True
def _print(*args, **kwargs):
    if is_print:
        print(*args, **kwargs)

rand_mat = None

cashed = {}


def flatten(x):
    return x.view(-1)


def make_dataloaders(params, raw=False, train=False, merge=False, permutate=False, train_size=1, map_dic=None):
    if map_dic is None:
        if merge:
            label_vals = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            shuffle(label_vals)
            map_dic = {k: v for k, v in zip(range(10), label_vals)}
        else:
            map_dic = {i: i for i in range(10)}

    if raw:
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_validation = transforms.Compose([transforms.ToTensor()])
    else:
        if train:
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        else:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                       [0.2023, 0.1994, 0.2010])])

        transform_validation = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                        [0.2023, 0.1994, 0.2010])])

    # def target_tranform(labels):
    if permutate:
        def randomize(_):
            random.randint(0, 9)
        target_tranform = np.vectorize(randomize)
    else:
        target_tranform = np.vectorize(map_dic.get)

    trainset = torchvision.datasets.CIFAR10(root=params['path'], train=True, transform=transform_train, target_transform=target_tranform, download=True)
    testset = torchvision.datasets.CIFAR10(root=params['path'], train=False, transform=transform_validation, target_transform=target_tranform)

    if train_size == 1:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params['batch_size'], shuffle=train)
        train_idxs = np.arange(len(trainset))
    else:
        train_idxs = np.random.permutation(np.arange(len(trainset)))[:int(len(trainset) * train_size)]
        train = [trainset[i] for i in train_idxs]

        trainloader = torch.utils.data.DataLoader(train, batch_size=params['batch_size'], shuffle=train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params['batch_size'], shuffle=False)
    return trainloader, testloader, (train_idxs, map_dic)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        ## CIFAR10
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # Stride 2 -> 1 for CIFAR-10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        ## END

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        # CIFAR10: stride 2 -> 1
        features = [ConvBNReLU(3, input_channel, stride=1)]
        # END

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def get_x(self, x):
        return x.view(x.size()[0], -1).cpu().numpy()

    def get_top_h(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return x.cpu().numpy()

    def get_all_h(self, x):
        tmp = x
        tmps = []
        for layer in self.features:
            tmp = layer(tmp)
            tmps.append(tmp.view(tmp.size()[0], -1))
        tmp = tmp.mean([2, 3])
        tmps.append(tmp.view(tmp.size()[0], -1))
        all_h = torch.cat(tmps, dim=1)
        return all_h.cpu().numpy()

    def get_flat_param(self):
        return torch.cat([flatten(p) for p in self.parameters()])

    def get_flat_param_grad(self):
        return torch.cat([flatten(p.grad) for p in self.parameters()])


    # def get_flat_param_grad_var(self):
    #     return torch.cat([flatten(p.grad_var) for p in self.parameters()])

    def get_grad_loss(self, test_data):
        criterion = nn.CrossEntropyLoss()
        image, label = test_data
        image = image.to(self.device)
        label = label.to(self.device)
        logit = self.forward(image)
        loss = criterion(logit, label)
        self.zero_grad()
        loss.backward()
        grad = self.get_flat_param_grad()
        return grad.cpu().numpy()

    def get_grad_output(self, test_data):
        image, label = test_data
        image = image.to(self.device)
        logit = self.forward(image)
        preds = nn.functional.softmax(logit)
        assert len(preds) == 1
        preds = preds[0]
        grads = []
        for p in preds:
            self.zero_grad()
            p.backward(retain_graph=True)
            grad = self.get_flat_param_grad()
            grads.append(grad.cpu().numpy())
        return np.vstack(grads)


    def get_influence_on_test_loss(self, test_data, train_data, test_description='',
        approx_type='cg', approx_params={}, force_refresh=False,
        matrix='none', sim_func=np.dot):

        self.train_data = train_data

        test_grad = self.get_grad_loss(test_data).astype(np.float32)

        if matrix == 'none':
            inverse_hvp = test_grad
        else:
            assert False

        start_time = time.time()

        predicted_loss_diffs = []
        train_grad_filename = os.path.join(self.out_dir, 'train-grad-on-loss-all.npy')
        if train_grad_filename in cashed:
            train_grads = cashed[train_grad_filename]
        elif os.path.exists(train_grad_filename):
            train_grads = np.load(train_grad_filename).astype(np.float32)
            _print('Loaded train grads from {}'.format(train_grad_filename))
        else:
            train_grad_list = []
            for counter, remove_data in enumerate(tqdm(train_data)):
                train_grad = self.get_grad_loss(remove_data)
                train_grad_list.append(train_grad.astype(np.float16))
            train_grads = np.vstack(train_grad_list)
            np.save(train_grad_filename, train_grads)
            _print('Saved train grads to {}'.format(train_grad_filename))

        if train_grad_filename not in cashed:
            cashed[train_grad_filename] = train_grads

        for counter, train_grad in enumerate(tqdm(train_grads)):
            predicted_loss_diffs.append(sim_func(inverse_hvp, train_grad))

        duration = time.time() - start_time
        _print('Multiplying by all train examples took {} sec'.format(duration))

        return predicted_loss_diffs


    def get_influence_by_hsim(self, test_data, train_data, h_fn, batch_size=100, sim_func=np.dot):
        with torch.no_grad():
            if h_fn == 'top_h':
                h_func = self.get_top_h
            elif h_fn == 'all_h':
                h_func = self.get_all_h
            elif h_fn == 'x':
                h_func = self.get_x
            else:
                assert False

            image, label = test_data
            image = image.to(self.device)
            test_feature = h_func(image).astype(np.float32)[0]

            feature_sims = []
            train_feature_filename = os.path.join(self.out_dir, 'train-{}-all.npy'.format(h_fn))

            if train_feature_filename in cashed:
                train_features = cashed[train_feature_filename]
            elif os.path.exists(train_feature_filename):
                train_features = np.load(train_feature_filename).astype(np.float32)
                _print('Loaded train features from {}'.format(train_feature_filename))
            else:
                train_feature_list = []
                for batch in tqdm(train_data):
                    image, label = batch
                    image = image.to(self.device)
                    features = h_func(image).astype(np.float32)
                    train_feature_list.append(features)
                train_features = np.vstack(train_feature_list)
                np.save(train_feature_filename, train_features)
                _print('Saved train features to {}'.format(train_feature_filename))
            if train_feature_filename not in cashed:
                cashed[train_feature_filename] = train_features

            for counter, train_feature in enumerate(tqdm(train_features)):
                feature_sims.append(sim_func(test_feature, train_feature))

            return feature_sims


def mobilenet_v2(pretrained=False, progress=True, device='cpu', model_path='', **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    return model


def test_model(model, params):
    model = model.to(params['device']).eval()
    phase = 'validation'
    logs = {'Accuracy': 0.0}

    # Iterate over data
    for image, label in pbar(params[phase + '_loader']):
        image = image.to(params['device'])
        label = label.to(params['device'])

        prediction = model(image)
        accuracy = torch.sum(torch.max(prediction, 1)[1] == label.data).item()
        logs['Accuracy'] += accuracy

    logs['Accuracy'] /= len(params[phase + '_loader'].dataset)

    return logs['Accuracy']
