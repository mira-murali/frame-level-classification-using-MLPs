import numpy as np
import os
import sys
import argparse

"""
Used the following code to pad the data and map labels to the corresponding frames
Default setting:
Pads the data given the default window size (10) and converts it into a single 2D array; saves this array
Stacks all the labels from each utterance into a single 2D array
Also normalizes the padded data and saves it separately
Stores the indices of the original frames after padding to be used in the dataloader
Assumes current location contains the data folder, i.e., WSJ_PATH is set to ./data/
To have no padding, simply set --kvalue 0
"""
parser = argparse.ArgumentParser(description='11785 HW1 Part 2 Data Padding')
parser.add_argument('--kvalue', default=10, type=int, help='Context window size')
parser.add_argument('--norm', default=1, type=int, help='Option to normalize data')
parser.add_argument('--std', default=0, type=int, help='Option to standardize data')
parser.add_argument('--ispadded', default=0, type=int, help='True if data given is already padded')
parser.add_argument('--islabeled', default=0, type=int, help='True if the frame levels labels have been stacked together')
parser.add_argument('--wsj_path', default='../data/', help='WSJ_PATH to set')
parser.add_argument('--isnoisy', default=0, type=int, help='True if noisy data already exists')
args = parser.parse_args()
class WSJ():
    """ Load the WSJ speech dataset

        Ensure WSJ_PATH is path to directory containing
        all data files (.npy) provided on Kaggle.

        Example usage:
            loader = WSJ()
            trainX, trainY = loader.train
            assert(trainX.shape[0] == 24590)

    """

    def __init__(self):
        self.dev_set = None
        self.train_set = None
        self.test_set = None

    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_raw(os.environ['WSJ_PATH'], 'dev')
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(os.environ['WSJ_PATH'], 'train')
        return self.train_set

    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(os.environ['WSJ_PATH'], 'test.npy'), encoding='bytes'), None)
        return self.test_set

def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes'),
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes')
    )

def pad_data(data, k, numfeatures, name):
    padded = []
    for i in range(data.shape[0]):
        print(i)
        new_pad = np.pad(data[i], ((k, k), (0, 0)), 'constant')
        padded.append(new_pad)

    padded = np.concatenate(padded, axis=0)
    np.save(os.environ['WSJ_PATH']+str(name), padded)
    return padded

def norm_data(train, dev, test):
    maxval = np.max(train, axis=0)
    minval = np.min(train, axis=0)
    norm_train = (train-minval)/(maxval-minval)
    norm_dev = (dev-minval)/(maxval-minval)
    norm_test = (test-minval)/(maxval-minval)
    return norm_train, norm_dev, norm_test

def std_data(train, dev, test):
    mean = np.mean(train, axis=0)
    sd = np.std(train, axis=0)
    std_train = (train-mean)/sd
    std_dev = (dev-mean)/sd
    std_test = (test-mean)/sd
    return std_train, std_dev, std_test

def stack_labels(labels, name):
    stack = labels[0].copy()
    for i in range(1, labels.shape[0]):
        stack = np.hstack((stack, labels[i]))

    np.save(os.environ['WSJ_PATH']+str(name), stack)
    return stack


def create_dict(data):
    ################ CODE FOR GETTING DICT FOR PADDED DATA ################
    indices = np.where(data.any(axis=1))[0]
    map_to_padded = np.zeros_like(indices)
    map_to_padded = indices.copy()
    return map_to_padded

def add_noise(data, norm):
    print("Adding noise")
    gauss = np.random.normal(0, 1, data.shape)
    new_data = gauss + data
    if norm:
        new_data[new_data>1] = 1
        new_data[new_data<0] = 0
    return new_data

def main():
    os.environ['WSJ_PATH'] = args.wsj_path
    loader = WSJ()
    trainX, trainY = loader.train
    devX, devY = loader.dev
    testX, testY = loader.test
    k = args.kvalue
    print("Loaded data")

    if not args.ispadded:
        padded_train = pad_data(trainX, k, 40, 'paddedtrain_'+str(k)+'.npy')
        padded_dev = pad_data(devX, k, 40, 'paddeddev_'+str(k)+'.npy')
        padded_test = pad_data(testX, k, 40, 'paddedtest_'+str(k)+'.npy')
        print(padded_test)
        train_map = create_dict(padded_train)
        dev_map = create_dict(padded_dev)
        test_map = create_dict(padded_test)

        np.save(os.environ['WSJ_PATH']+'train_dict_'+str(k)+'.npy', train_map)
        np.save(os.environ['WSJ_PATH']+'dev_dict_'+str(k)+'.npy', dev_map)
        np.save(os.environ['WSJ_PATH']+'test_dict_'+str(k)+'.npy', test_map)

    else:

        padded_train = np.load(os.environ['WSJ_PATH']+'paddedtrain_'+str(k)+'.npy')
        padded_dev = np.load(os.environ['WSJ_PATH']+'paddeddev_'+str(k)+'.npy')
        padded_test = np.load(os.environ['WSJ_PATH']+'paddedtest_'+str(k)+'.npy')

    print("Got paddded data")
    if not args.islabeled:
        train_dict_labels = stack_labels(trainY, 'train_dict_labels.npy')
        dev_dict_labels = stack_labels(devY, 'dev_dict_labels.npy')

    print("Got labeled data")
    if args.std:
        std_train, std_dev, std_test = std_data(padded_train, padded_dev, padded_test)
        np.save(os.environ['WSJ_PATH']+'paddedtrain_std_'+str(k)+'.npy', std_train)
        np.save(os.environ['WSJ_PATH']+'paddeddev_std_'+str(k)+'.npy', std_dev)
        np.save(os.environ['WSJ_PATH']+'paddedtest_std_'+str(k)+'.npy', std_test)
        if not args.isnoisy:
            noisy = add_noise(std_train, norm=False)
            np.save(os.environ['WSJ_PATH']+'paddedtrain_std_noisy_'+str(k)+'.npy', noisy)

    else:
        if not args.isnoisy:
            std_train = np.load(os.environ['WSJ_PATH']+'paddedtrain_std_'+str(k)+'.npy', encoding='bytes')
            noisy = add_noise(std_train, norm=False)
            np.save(os.environ['WSJ_PATH']+'paddedtrain_std_noisy_'+str(k)+'.npy', noisy)

    print("Got std and noisy data")


    if args.norm:
        norm_train, norm_dev, norm_test = norm_data(padded_train, padded_dev, padded_test)
        np.save(os.environ['WSJ_PATH']+'paddedtrain_norm_'+str(k)+'.npy', norm_train)
        np.save(os.environ['WSJ_PATH']+'paddeddev_norm_'+str(k)+'.npy', norm_dev)
        np.save(os.environ['WSJ_PATH']+'paddedtest_norm_'+str(k)+'.npy', norm_test)
        if not args.isnoisy:
            noisy = add_noise(norm_train, norm=True)
            np.save(os.environ['WSJ_PATH']+'paddedtrain_norm_noisy_'+str(k)+'.npy', noisy)
    else:
        if not args.isnoisy:
            norm_train = np.load(os.environ['WSJ_PATH']+'paddedtrain_norm_'+str(k)+'.npy', encoding='bytes')
            noisy = add_noise(norm_train, norm=True)
            np.save(os.environ['WSJ_PATH']+'paddedtrain_norm_noisy_'+str(k)+'.npy', noisy)

    print("Got norm and noisy data")







if __name__ == '__main__':
    main()
