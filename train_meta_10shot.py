import torch
import numpy as np
from GenTask import GenTask
import argparse
from torch.utils.data import DataLoader

# from torch.optim import lr_scheduler
# import random
# import sys
# import pickle
# import os

from meta import Meta

def main():

    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    np.random.seed(2024)

    print(args)

    config = [
        ('complex_conv', [64, 1, 3, 1, 0]),  # block1
        ('relu', [True]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block2
        ('relu', [True]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block3
        ('relu', [True]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block4
        ('relu', [True]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block5
        ('relu', [True]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block6
        ('relu', [True]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block7
        ('relu', [True]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block8
        ('relu', [True]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('complex_conv', [64, 64, 3, 1, 0]),  # block9
        ('relu', [True]),
        ('bn', [128]),
        ('max_pool1d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [1024, 1152]),
        ('relu', [True]),
        ('linear', [args.n_way, 1024])
    ]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # generate meta-training tasks
    mini = GenTask('/data/fuxue/Dataset_WiFi_maml', mode='train', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, batch_size=100000)

    for epoch in range(args.epochs):
        # fetch meta_batch_size num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            acc = maml(args.weight_tri, x_spt, y_spt, x_qry, y_qry)

            if step % 10 == 0:
                torch.save(maml.state_dict(), args.save_path)
                print('epoch:', epoch, 'step:', step, '\ttraining acc:', acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='epoch number', default=3)
    parser.add_argument('--n_way', type=int, help='n way', default=6)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=6)  # 修改此部分可降低显卡占用
    parser.add_argument('--weight_tri', type=int, help='weight of triplet loss in sum loss', default=0.01)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=2)
    parser.add_argument('--update_step_test', type=int, help='update steps for fine_tuning', default=10)
    parser.add_argument('--save_path', type=str, help='path for saving model weight', default='model_weight/cvnn_maml-st_6way_10shot.pth')

    args = parser.parse_args()

    main()
