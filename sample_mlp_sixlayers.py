import argparse
import os
import shutil
import time
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import spline
# import matplotlib.patches as mpatches
# from dataloader import *
from dataloader_noisy import *
# from HW1.wsj_loader import *
"""
To run this code from the terminal:
python sample_mlp_sixlayers --numfeatures 1024 1024 1024 1024 1024 1024 1024 or any other preferable number of hidden units per layer
Default setting:
window size: 10
epochs: 15
learning rate: 0.1, decreased to 0.01 after half the number of epochs
normalized data (could be changed to load in standardized data instead using argument--std 1 and --norm 0)
batchnorm momentum value: 0.008
Note: Code will not run unless --numfeatures is explicitly specified
"""

parser = argparse.ArgumentParser(description='11785 HW1 Part 2')
parser.add_argument('--window', default=10, type=int, help='--k value for context window')
parser.add_argument('--epochs', default=15, type=int, help='--number of epochs to train for')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--dampening', '--damp', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--batch_size', '--bs', default = 512, type = int, help='batch size for data')
parser.add_argument('--lr', default=0.1, type = float, help='learning rate for gradient descent')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--print_freq', default=2000, type=int, help='how often to print accuracy')
# parser.add_argument('--fcnum1', default=4096, type=int, help='first hidden layer units')
# parser.add_argument('--fcnum2', default=4096, type=int, help='second hidden layer units')
# parser.add_argument('--fcnum3', default=4096, type=int, help='third hidden layer units')
parser.add_argument('--numfeatures', nargs='+', type=int, help='list of feature numbers')
parser.add_argument('--subfilename', default='sample_submission.csv', type=str, help='third hidden layer units')
parser.add_argument('--bnmomentum', default = 0.008, type =float, help ='batchnorm momentum value')
parser.add_argument('--adjust', default = 1, type = int, help='decrease learning rate or no')
parser.add_argument('--norm', default=0, type=int, help='add the normalized data')
parser.add_argument('--noisy', default=0, type=int, help='add gaussian noise to data')
parser.add_argument('--best_model', default=1, type=int, help='make predictions based on best validation model')
parser.add_argument('--std', default=1, type=int, help='standardized data')
# parser.add_argument('--context', default=1, type=int, help='Assumes context window is required')
# parser.add_argument('--pause', default=0, type=int, help='pause in the middle of epochs to exit')

args = parser.parse_args()

class customMLP(nn.Module):

    def __init__(self, input_size, num_features, num_classes= 138):
        super(customMLP, self).__init__()

        self.layers = nn.Sequential(nn.Linear(input_size, num_features[0]),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm1d(num_features[0], momentum=args.bnmomentum),
                                    nn.Linear(num_features[0], num_features[1]),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm1d(num_features[1], momentum = args.bnmomentum),
                                    nn.Linear(num_features[1], num_features[2]),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm1d(num_features[2], momentum = args.bnmomentum),
                                    # nn.Dropout(0.2),
                                    nn.Linear(num_features[2], num_features[3]),
			                        nn.LeakyReLU(),
                                    nn.BatchNorm1d(num_features[3], momentum=args.bnmomentum),
                                    nn.Dropout(0.2),
				                    nn.Linear(num_features[3], num_features[4]),
				                    nn.LeakyReLU(),
				                    nn.BatchNorm1d(num_features[4], momentum=args.bnmomentum),
                                    nn.Dropout(0.2),
				                    nn.Linear(num_features[4], num_features[5]),
				                    nn.LeakyReLU(),
                                    nn.Dropout(0.2),
				                    nn.Linear(num_features[5], num_classes)
                                    )

    def forward(self, x):
        y = self.layers(x)
        # print(y.shape)
        return y


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    global args
    criterion = nn.CrossEntropyLoss().to(device)
    features = args.numfeatures
    model = customMLP(input_size = 40*((2*args.window)+1), num_features=features)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum, args.dampening)
    # print(optimizer)
    # sys.exit(0)

    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,                                   # torch.distributed package, for parallel running on multi-nodes on multi-computer
                                world_size=args.world_size)


    if args.std:
        if args.noisy:
            train_dataset = DataLoaderFrame(root_dir='./data/', window=args.window, frame_name='paddedtrain_std_'+str(args.window)+'.npy', dict_name='train_dict_'+str(args.window)+'.npy',
                                             label_name='train_dict_labels.npy', noisy='paddedtrain_std_noisy_'+str(args.window)+'.npy')
        else:
            train_dataset = DataLoaderFrame(root_dir='./data/', window=args.window, frame_name='paddedtrain_std_'+str(args.window)+'.npy', dict_name='train_dict_'+str(args.window)+'.npy',
                                             label_name='train_dict_labels.npy', noisy=None)
        dev_dataset = DataLoaderFrame(root_dir='./data/', window=args.window, frame_name='paddeddev_std_'+str(args.window)+'.npy', dict_name='dev_dict_'+str(args.window)+'.npy',
                                         label_name='dev_dict_labels.npy', noisy=None)
        test_dataset = DataLoaderFrame(root_dir = './data/', window = args.window, frame_name = 'paddedtest_std_'+str(args.window)+'.npy', dict_name='test_dict_'+str(args.window)+'.npy',
                                         label_name = None, noisy=None)
    if args.norm:
        if args.noisy:
            train_dataset = DataLoaderFrame(root_dir='./data/', window=args.window, frame_name='paddedtrain_norm_'+str(args.window)+'.npy', dict_name='train_dict_'+str(args.window)+'.npy',
                                             label_name='train_dict_labels.npy', noisy='paddedtrain_norm_noisy_'+str(args.window)+'.npy')
        else:
            train_dataset = DataLoaderFrame(root_dir='./data/', window=args.window, frame_name='paddedtrain_norm_'+str(args.window)+'.npy', dict_name='train_dict_'+str(args.window)+'.npy',
                                             label_name='train_dict_labels.npy', noisy=None)
        dev_dataset = DataLoaderFrame(root_dir='./data/', window=args.window, frame_name='paddeddev_norm_'+str(args.window)+'.npy', dict_name='dev_dict_'+str(args.window)+'.npy',
                                         label_name='dev_dict_labels.npy', noisy=None)
        test_dataset = DataLoaderFrame(root_dir = './data/', window = args.window, frame_name = 'paddedtest_norm_'+str(args.window)+'.npy', dict_name='test_dict_'+str(args.window)+'.npy',
                                         label_name = None, noisy=None)




    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


    n_epochs = args.epochs
    training_accuracy = np.zeros(n_epochs)
    developing_accuracy = np.zeros(n_epochs)
    training_losses = np.zeros(n_epochs)
    developing_losses = np.zeros(n_epochs)
    isbest_train = False
    isbest_dev = False
    max_train = 0
    max_dev = 0

    print(model)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None


    train_predictions = None
    dev_predictions = None
    test_predictions = None
    best_model = None
    for epoch in range(n_epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.adjust:
            adjust_learning_rate(optimizer, epoch)

        train_loss, train_accuracy = train(model, criterion, optimizer, train_loader, epoch)
        dev_loss, dev_accuracy = develop(model, criterion, dev_loader, epoch)
        if max_train < train_accuracy:
            max_train = train_accuracy

        if max_dev < dev_accuracy:
            max_dev = dev_accuracy
            isbest_dev = True
        if isbest_dev:
            isbest_dev = False
            save_checkpoint(model, "model_best.pth.tar")
            best_model = model

        training_losses[epoch] = train_loss
        training_accuracy[epoch] = train_accuracy

        developing_losses[epoch] = dev_loss
        developing_accuracy[epoch] = dev_accuracy

        print("Highest Training Accuracy: ", max_train)
        print("Highest Validation Accuracy: ", max_dev)



    print("Training Accuracy per epoch: ", training_accuracy)
    print("Training Loss per epoch: ", training_losses)
    print("Validation Accuracy per epoch: ", developing_accuracy)
    print("Validation Loss per epoch: ", developing_losses)

    if args.best_model:
        test_predictions = test(best_model, test_loader)
        write_to_csv(test_predictions, args.subfilename)
    else:
        test_predictions = test(model, test_loader)
        write_to_csv(test_predictions, args.subfilename)


def add_noise(x):
    gauss = np.random.normal(0, 1, x.shape)
    gauss = torch.from_numpy(gauss)
    gauss = gauss.float()
    return x + gauss

def scale(x, up=True):
    if up:
        return x*2
    else:
        return x/2

def data_std(x):
    mean = x.mean(dim= 0)
    std = x.std(dim= 0)
    std_x = (x - mean)/std
    return std_x

def train(model, criterion, optimizer, train_loader, epoch):

    model.train()
    train_loss = 0
    train_acc = 0
    all_pred = None
    total = 0
    for i, (frames, labels) in enumerate(train_loader):

        frames, labels = frames.to(device), labels.to(device)
        # print(labels)
        output = model(frames)
        loss = criterion(output, labels)
        acc, predictions = accuracy(output, labels, True)

      #  if i == 0:
       #     all_pred = predictions
       # else:
       #     all_pred = torch.cat((all_pred, predictions))

        train_loss += loss.item()
        train_acc += acc
        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%args.print_freq==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {3}\t'
                  'Accuracy {4}'.format(
                   epoch, i, len(train_loader), loss, acc))


    total = len(train_loader)
    return train_loss/total, train_acc/total

def develop(model, criterion, dev_loader, epoch):

    model.eval()
    dev_loss = 0
    dev_acc = 0
    all_pred = None
    total = 0
    with torch.no_grad():
        for i, (frames, labels) in enumerate(dev_loader):
            frames, labels = frames.to(device), labels.to(device)
            output = model(frames)
            loss = criterion(output, labels)

            acc, predictions = accuracy(output, labels, True)
            # if i == 0:
              #  all_pred = predictions
            # else:
              #  all_pred = torch.cat((all_pred, predictions))

            dev_loss += loss.item()
            dev_acc += acc
            total += labels.size(0)
            if i%args.print_freq==0:
                print('Test: [{0}/{1}]\t'
                          'Loss {2}\t'
                          'Accuracy {3}'.format(
                           i, len(dev_loader), loss,
                           acc))


    total = len(dev_loader)
    return dev_loss/total, dev_acc/total

def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_pred = None
    with torch.no_grad():
        for i, (frames, _) in enumerate(test_loader):
            frames = frames.to(device)
            output = model(frames)
            _, pred = accuracy(output, label=False)

            if i==0:
                test_pred = pred
            else:
                test_pred = torch.cat((test_pred, pred))

            if i%args.print_freq==0:
                print('Test: [{0}/{1}]'.format(i, len(test_loader)))

    return test_pred

def accuracy(output, target=None, label=True):
    """Computes the precision@k for the specified values of k"""
    correct = 0
    with torch.no_grad():
        batch_size = output.size(0)

        _, pred = torch.max(output, 1)

        #print(pred.size())
        if label:
            correct += (pred==target).sum().item()
            correct = (correct/batch_size)*100

            return correct, pred
        else:
            return None, pred
        #print(target.size())


def adjust_learning_rate(optimizer, epoch):
    global args
    lr = args.lr
    if epoch >= args.epochs/2:
    	lr  *= 0.1
    # elif epoch >= (2*args.epochs)/3:
    #     lr  *= 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def write_to_csv(predictions, filename):
    f = open(filename, "w")
    pred_writer = csv.writer(f, delimiter=',')
    pred_writer.writerow(['id', 'label'])
    for i in range(len(predictions)):
        pred_writer.writerow([str(i), str(predictions[i].item())])

def save_checkpoint(state, filepath):
        torch.save(state, filepath)

if __name__=='__main__':
    main()
