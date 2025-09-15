import torch
import numpy as np
import torch.nn.functional as F
import argparse
import os
import collections
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from get_dataset import load_pretrain_dataset, load_train_dataset_k_shot, load_test_dataset
from collections import OrderedDict
from learner import Learner
from torch import optim
from scipy.io import savemat

def train_dataset_prepared(n_classes):
    x_pretrain, value_y_pretrain = load_pretrain_dataset(10)
    min_value = x_pretrain.min()
    max_value = x_pretrain.max()
    x_train, value_y_train = load_train_dataset_k_shot(n_classes)
    x_pretrain = (x_pretrain - min_value) / (max_value - min_value)
    x_train = (x_train - min_value) / (max_value - min_value)
    return x_pretrain, x_train, value_y_pretrain, value_y_train

def test_dataset_prepared(n_classes):
    x_pretrain, value_y_pretrain = load_pretrain_dataset(10)
    min_value = x_pretrain.min()
    max_value = x_pretrain.max()
    x_test1, value_y_test1, x_test2, value_y_test2 = load_test_dataset(n_classes)
    x_test1 = (x_test1 - min_value) / (max_value - min_value)
    x_test2 = (x_test2 - min_value) / (max_value - min_value)
    return x_test1, x_test2, value_y_test1, value_y_test2

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
def extract_feature(val_loader, model, set='base'):
    save_dir = '../feature'
    if os.path.isfile(save_dir + './%s_features.plk' % set):
        data = load_pickle(save_dir + './%s_features.plk' % set)
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():

        output_dict = collections.defaultdict(list)

        for i, (inputs, labels) in enumerate(val_loader):
            # compute output
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs, _ = model(inputs)
            outputs = outputs.cpu().data.numpy()
            labels = labels.cpu().data.numpy()

            features_list.append(outputs)
            labels_list.append(labels)
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)

        all_info = output_dict
        save_pickle(save_dir + './%s_features.plk' % set, all_info)

        mat_features = {'features': np.concatenate(features_list)}
        mat_labels = {'labels': np.concatenate(labels_list)}
        savemat(save_dir + './%s_features.mat' % set, mat_features)
        savemat(save_dir + './%s_labels.mat' % set, mat_labels)

        return all_info

class Config:
    def __init__(
        self,
        train_batch_size: int = 16,
        test_batch_size: int = 16,
        epochs: int = 300,
        lr: float = 0.01,
        n_classes: int = 6,
        pretrain_path: str = "../model_weight/cvnn_maml-st_6way_10shot.pth",
    ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.n_classes = n_classes
        self.pretrain_path = pretrain_path

if __name__ == '__main__':
    conf = Config()

    x_pretrain, x_train, value_y_pretrain, value_y_train = train_dataset_prepared(conf.n_classes)

    pretrain_dataset = TensorDataset(torch.Tensor(x_pretrain), torch.Tensor(value_y_pretrain))
    pretrain_data_loader = DataLoader(pretrain_dataset, batch_size=conf.train_batch_size, shuffle=True)

    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(value_y_train))
    train_data_loader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True)

    x_test1, x_test2, value_y_test1, value_y_test2 = test_dataset_prepared(conf.n_classes)

    test_dataset1 = TensorDataset(torch.Tensor(x_test1), torch.Tensor(value_y_test1))
    test_data_loader1 = DataLoader(test_dataset1, batch_size=conf.test_batch_size, shuffle=True)

    test_dataset2 = TensorDataset(torch.Tensor(x_test2), torch.Tensor(value_y_test2))
    test_data_loader2 = DataLoader(test_dataset2, batch_size=conf.test_batch_size, shuffle=True)

    config = [
        ('complex_conv', [64, 1, 3, 1, 0]),  # block1
        ('relu', [False]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block2
        ('relu', [False]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block3
        ('relu', [False]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block4
        ('relu', [False]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block5
        ('relu', [False]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block6
        ('relu', [False]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block7
        ('relu', [False]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block8
        ('relu', [False]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block9
        ('relu', [False]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [1024, 1152]),
        ('relu', [False]),
        ('linear', [conf.n_classes, 1024])
    ]

    model = Learner(config)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(conf.pretrain_path, map_location='cuda')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[4:]  # remove 'net.'
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    output_dict_base = extract_feature(pretrain_data_loader, model, set='base')
    print("base set features saved!")

    output_dict_novel1 = extract_feature(train_data_loader, model, set='novel1')
    print("novel features saved!")

    output_dict_qry1 = extract_feature(test_data_loader1, model, set='query1')
    print("qry1 features saved!")

    output_dict_qry2 = extract_feature(test_data_loader2, model, set='query2')
    print("qry2 features saved!")
