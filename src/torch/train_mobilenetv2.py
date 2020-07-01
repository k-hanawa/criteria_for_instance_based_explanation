import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm as pbar
from torch.utils.tensorboard import SummaryWriter
from src.torch import mobilenetv2
import argparse
import os
import random
import numpy as np
import pickle
import json

def train_model(model, params, summary_path, output):
    writer = SummaryWriter(summary_path)
    model = model.to(params['device'])
    optimizer = optim.AdamW(model.parameters())
    total_updates = params['num_epochs'] * len(params['train_loader'])

    criterion = nn.CrossEntropyLoss()
    best_accuracy = test_model(model, params)
    best_model = copy.deepcopy(model.state_dict())

    for epoch in pbar(range(params['num_epochs'])):
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:

            # Loss accumulator for each epoch
            logs = {'Loss': 0.0,
                    'Accuracy': 0.0}

            # Set the model to the correct phase
            model.train() if phase == 'train' else model.eval()

            # Iterate over data
            for image, label in params[phase + '_loader']:
                image = image.to(params['device'])
                label = label.to(params['device'])

                # Zero gradient
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # Forward pass
                    prediction = model(image)
                    loss = criterion(prediction, label)
                    accuracy = torch.sum(torch.max(prediction, 1)[1] == label.data).item()

                    # Update log
                    logs['Loss'] += image.shape[0] * loss.detach().item()
                    logs['Accuracy'] += accuracy

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            # Normalize and write the data to TensorBoard
            logs['Loss'] /= len(params[phase + '_loader'].dataset)
            logs['Accuracy'] /= len(params[phase + '_loader'].dataset)
            writer.add_scalars('Loss', {phase: logs['Loss']}, epoch)
            writer.add_scalars('Accuracy', {phase: logs['Accuracy']}, epoch)

            # Save the best weights
            if phase == 'validation' and logs['Accuracy'] > best_accuracy:
                best_accuracy = logs['Accuracy']
                best_model = copy.deepcopy(model.state_dict())

        # Write best weights to disk
        if epoch % params['check_point'] == 0 or epoch == params['num_epochs'] - 1:
            torch.save(best_model, output)

    final_accuracy = test_model(model, params)
    writer.add_text('Final_Accuracy', str(final_accuracy), 0)
    writer.close()


def test_model(model, params):
    model = model.to(params['device']).eval()
    phase = 'validation'
    logs = {'Accuracy': 0.0}

    # Iterate over data
    for image, label in pbar(params[phase + '_loader']):
        image = image.to(params['device'])
        label = label.to(params['device'])

        with torch.no_grad():
            prediction = model(image)
            accuracy = torch.sum(torch.max(prediction, 1)[1] == label.data).item()
            logs['Accuracy'] += accuracy

    logs['Accuracy'] /= len(params[phase + '_loader'].dataset)

    return logs['Accuracy']

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batchsize', '-b', type=int, default=256)
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--pretrained_model', default=None)
    parser.add_argument('--train_size', type=float, default=1)
    parser.add_argument('--merge_label', action='store_true')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='result')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Train on cuda if available
    if args.gpu > -1:
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    data_params = {'path': 'data/cifar10_torchvision', 'batch_size': 256}

    train_loader, validation_loader, (train_idxs, label_dic) = mobilenetv2.make_dataloaders(data_params, train=True, merge=args.merge_label, train_size=args.train_size)

    train_params = {'description': 'Test',
                    'num_epochs': args.epoch,
                    'check_point': 10, 'device': device,
                    'train_loader': train_loader,
                    'validation_loader': validation_loader}

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    with open(os.path.join(args.out, 'train_idxs.pkl'), 'wb') as fo:
        pickle.dump(train_idxs, fo)
    with open(os.path.join(args.out, 'label.json'), 'w') as fo:
        json.dump(label_dic, fo)


    if args.merge_label:
        n_class = 2
    else:
        n_class = 10
    if args.pretrained_model is not None:
        model = mobilenetv2.mobilenet_v2(pretrained=True, device=device, num_classes=n_class, model_path=args.pretrained_model)
    else:
        model = mobilenetv2.mobilenet_v2(device=device, num_classes=n_class)

    train_model(model, train_params, os.path.join(args.out, 'summary'), os.path.join(args.out, 'best_model.pt'))


if __name__ == '__main__':
    main()
