import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as dl

import numpy as np
import os
import sys

class DataLoaderFrame(dl.Dataset):

    def __init__(self, root_dir, window, frame_name, dict_name, label_name=None, noisy=None):
        self.root_dir = root_dir
        self.window = window
        self.dict_name = dict_name
        self.frame = np.load(self.root_dir+frame_name)

        self.map_to_dict = np.load(self.root_dir+self.dict_name)
        if label_name is not None:
            self.labels = np.load(self.root_dir+label_name)
        else:
            self.labels = None
        if noisy is not None:
            self.noisy = np.load(self.root_dir+noisy)
        else:
            self.noisy = None

        # self.transform = transform


    def __len__(self):
        return len(self.map_to_dict)

    def __getitem__(self, idx):
        # if self.context:
        index = self.map_to_dict[idx]
        if self.noisy is not None:
            g = np.random.uniform(0, 1)
            if g<0.5:
                frame = torch.from_numpy(self.noisy[index-self.window:index+self.window+1])
                frame = frame.float()
            else:
                frame = torch.from_numpy(self.frame[index-self.window:index+self.window+1])
        else:
            frame = torch.from_numpy(self.frame[index-self.window:index+self.window+1])

        frame = frame.view(frame.shape[0]*frame.shape[1])

        # else:
        #     if self.noisy is not None:
        #         g = np.random.uniform(0, 1)
        #         if g<0.5:
        #             frame = torch.from_numpy(self.noisy[idx])
        #             frame = frame.float()
        #         else:
        #             frame = torch.from_numpy(self.frame[idx])
        #     else:
        #         frame = torch.from_numpy(self.frame[idx])

        if self.labels is not None:
            labels = torch.LongTensor([self.labels[idx]])
            labels = labels.item()
            return (frame, labels)
        else:
            labels = torch.LongTensor([0])
            labels = labels.item()

            return (frame, labels)



        # labels[self.labels[idx]] = 1
        # labels = labels.view(1, labels.shape[0])
        # print(frame.shape, labels.shape)
        # frame = frame.unsqueeze(dim=0)
        # # frame = frame.view(-1, )
        # labels = labels.unsqueeze(dim=0)
        #
        # frame = frame.view(1, frame.shape[1]*frame.shape[2])
        # labels = labels.unsqueeze(dim=0)
