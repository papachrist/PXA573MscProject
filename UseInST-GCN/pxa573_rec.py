#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
# from processor.Qualysis import readClasses
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

outfile = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/topk/'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def readClasses():
    import csv
    classes = []
    with open('/media/papachristo/My Passport/finalp/Classes.csv', mode='r') as infile:
        reader = csv.reader(infile)

        for rows in reader:
            k = rows
            # v = rows[1]
            classes.append(k)
    return classes

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """
    def loadmydata(self):

        direct = '/media/papachristo/My Passport/finalp/MyQualysis/Out/'
        titles = ['S01P1', 'S01P4', 'S02P2', 'S02P4'] #['S01P1', 'S01P2', 'S01P3', 'S01P4', 'evalS01P5', 'S02P1', 'S02P2', 'S02P3', 'S02P4', 'S02P5', 'evalS02P6', 'evalS02P7']
        loadata = []
        for t in titles:
            data = direct + t +'.npy'
            Labels = direct + 'L' + t

            # loader = self.data_loader['test']
            pkl_file = open(Labels, 'rb')
            labels = pickle.load(pkl_file)
            pkl_file.close()

            valdata = np.load(data)

            for ind in range(len(valdata)):
                #include batch
                print(labels[0][ind])
                loadata.append([torch.from_numpy(np.asarray([valdata[ind]])), labels[1][ind]])

        return loadata

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            # UNTIL HERE HAS TO BE THE SAME LIKE TEST
            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']

        loss_value = []
        result_frag = []
        label_frag = []
        classes = readClasses()
        cntr = 0
        tot = len(loader)
        toparr = []
        for data, label in loader:

        # get data
            batch = data.shape[0]
            cntr += 1
            if (cntr % 10) == 0 and batch > 1:
                print("{} {} / {}".format("Evaluated: ", cntr*batch, tot*batch))
            data = data.float().to(self.dev)
            # print(data.cpu().numpy().shape)
            label = label.long().to(self.dev)
            # inference
            with torch.no_grad():
                output = self.model(data)

            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                ## FOR LIVE EVALUATION ##
                if batch == 1:
                    OutArr = output.cpu().numpy()
                    # print(OutArr)
                    print(cntr-1)
                    prediction = np.argmax(OutArr)
                    print("{} {}".format("Expected: ", classes[label[0]]))
                    Sorted = np.argsort(OutArr)
                    print("{} {} {}".format("Top Prediction: ", classes[prediction], OutArr[0][Sorted[0][59]]))
                    print("{} {} {}".format("2nd Prediction: ", classes[Sorted[0][58]], OutArr[0][Sorted[0][58]]))
                    print("{} {} {}".format("3rd Prediction: ", classes[Sorted[0][57]], OutArr[0][Sorted[0][57]]))
                    print("{} {} {}".format("4th Prediction: ", classes[Sorted[0][56]], OutArr[0][Sorted[0][56]]))
                    print("{} {} {}".format("5th Prediction: ", classes[Sorted[0][55]], OutArr[0][Sorted[0][55]]))
                    # print(Sorted)
                    topk = Sorted[0][55:]
                    if int(label[0]) not in topk:
                        toparr.append(100)
                    else:
                        for tind in range(len(topk)):
                            if int(label[0]) == topk[tind]:
                                toparr.append(len(topk)-tind)

                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        if batch == 1:
            sve = input('Save top_array [y/n]? ')
            if sve == 'y':
                name = input('File name:')
                np.save(outfile + 'topk' + name, np.asarray(toparr))
        self.result = np.concatenate(result_frag)

        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
