import os
import torch
import numpy as np
import csv
import random
from torch.utils.data import Dataset

from torchvision.transforms import transforms
import collections
from PIL import Image


def load_csv(csv_filename):
    """
    return a dict saving the information of csv
    param csv_filename: csv file name
    return: {label:[file1, file2 ...]}
    """

    dict_labels = {}
    with open(csv_filename) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]
            # append filename to current label
            if label in dict_labels.keys():
                dict_labels[label].append(filename)
            else:
                dict_labels[label] = [filename]
    return dict_labels


def load_npy(path):
    x = np.load(path)
    x = torch.Tensor(x)
    return x


class GenTask(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all images
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several set
    set: contains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batch_size, n_way, k_shot, k_query, start_idx=0):
        """
        def __init__(self, root, mode, batch_size, n_way, k_shot, k_query, resize, start_idx=0):
        param root: root path of mini-imagenet
        param mode: train, val or test
        param batch_size: batch size of sets/tasks, not batch of images
        param n_way: num of classes
        param k_shot: num of support images per class
        param k_query: num of query images per class
        param resize: resize to (This parameter is not required for signal datasets)
        param start_idx: start to index label from start_idx
        """

        self.batch_size = batch_size  # batch of sets/tasks, not batch of images
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.start_idx = start_idx  # index label not from 0, but from start_idx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query' % (mode, batch_size, n_way, k_shot, k_query))

        # for create_batch
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch

        self.path = os.path.join(root, 'data')  # image path
        csv_data = load_csv(os.path.join(root, mode + '.csv'))  # csv path
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csv_data.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[k] = i + self.start_idx  # {"img_name[:9]":label}
        self.cls_num = len(self.data)

        self.create_batch(self.batch_size)

    def create_batch(self, batch_size):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        param episodes: batch size
        :return:
        """

        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batch_size):  # for each batch/task
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_images_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_images_idx)
                index_train_d = np.array(selected_images_idx[:self.k_shot])  # idx for Data_train
                index_test_d = np.array(selected_images_idx[self.k_shot:])  # idx for Dtest
                support_x.append(np.array(self.data[cls])[index_train_d].tolist())  # get all images filename for current Data_train
                query_x.append(np.array(self.data[cls])[index_test_d].tolist())

            # shuffle the corresponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batch_size-1
        :param index:
        :return:
        """

        # support_x = torch.FloatTensor(self.set_size, 3, self.resize, self.resize)  # [set_size, 3, resize, resize]
        support_x = torch.FloatTensor(self.set_size, 2, 6000)  # support_x = torch.FloatTensor(self.set_size, 2, 6000): [set_size, 2, 6000]
        support_y = np.zeros(self.set_size, dtype=np.int)  # [set_size]

        # query_x = torch.FloatTensor(self.query_size, 3, self.resize, self.resize)  # [query_size, 3, resize, resize]
        query_x = torch.FloatTensor(self.query_size, 2, 6000)  # query_x = torch.FloatTensor(self.query_size, 2, 6000):[query_size, 2, 6000]
        query_y = np.zeros(self.query_size, dtype=np.int)  # [query_size]

        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array([self.img2label[item[:5]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
                              for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:5]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        # print('global:', support_y, query_y)  # support_y: [set_size]  # query_y: [query_size]

        unique = np.unique(support_y)  # unique: [n-way], sorted
        unique = list(unique)
        random.shuffle(unique)
        unique = np.array(unique)

        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.set_size)
        query_y_relative = np.zeros(self.query_size)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx
        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            support_x[i] = load_npy(path)  # support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = load_npy(path)  # query_x[i] = self.transform(path)
        # print(support_set_y)

        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)
        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):  # as we have built up to batch_size of sets, you can sample some small batch size of sets.
        return self.batch_size


def main():
    pass


if __name__ == '__main__':
    main()
