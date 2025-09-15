import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from TripletLoss import TripletLoss
from learner import Learner
from copy import deepcopy

def clip_grad_by_norm_(grad, max_norm):
    """
    in-place gradient clipping.
    param grad: list of gradients
    param max_norm: maximum norm allowable
    return:
    """
    total_norm = 0
    counter = 0

    for g in grad:
        param_norm = g.data.norm(2)
        total_norm += param_norm.item() ** 2
        counter += 1
    total_norm = total_norm ** (1. / 2)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grad:
            g.data.mul_(clip_coef)
    return total_norm/counter


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config)  # self.net = Learner(config, args.imgc, args.image_size)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.triplet_loss = TripletLoss(margin=5)

    def forward(self, weight_tri, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, set_size, c_, h, w]
        :param y_spt:   [b, set_size]
        :param x_qry:   [b, query_size, c_, h, w]
        :param y_qry:   [b, query_size]
        :return:
        """
        task_num, set_size, h, w = x_spt.size()  # task_num, set_size, c_, h, w = x_spt.size()  # [task_num, set_size, 2, 6000]
        query_size = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            output = self.net(x_spt[i], net_vars=None, bn_training=True)
            embedding, logic = output[0], output[1]
            loss_ce = F.cross_entropy(logic, y_spt[i])  # compute CE loss
            loss_triplet = self.triplet_loss(embedding, y_spt[i])  # compute triplet loss
            loss = loss_ce + weight_tri * loss_triplet
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [set_size, n_way]
                output_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                embedding_q, logic_q = output_q[0], output_q[1]
                loss_ce_q = F.cross_entropy(logic_q, y_qry[i])
                loss_triplet_q = self.triplet_loss(embedding_q, y_qry[i])
                loss_q = loss_ce_q + weight_tri * loss_triplet_q
                losses_q[0] += loss_q

                predict_q = F.softmax(logic_q, dim=1).argmax(dim=1)
                correct = torch.eq(predict_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [set_size, n_way]
                output_q = self.net(x_qry[i], fast_weights, bn_training=True)
                embedding_q, logic_q = output_q[0], output_q[1]
                loss_ce_q = F.cross_entropy(logic_q, y_qry[i])
                loss_triplet_q = self.triplet_loss(embedding_q, y_qry[i])
                loss_q = loss_ce_q + weight_tri * loss_triplet_q
                losses_q[1] += loss_q
                # [set_size]
                predict_q = F.softmax(logic_q, dim=1).argmax(dim=1)
                correct = torch.eq(predict_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                output = self.net(x_spt[i], fast_weights, bn_training=True)
                embedding, logic = output[0], output[1]
                loss_ce = F.cross_entropy(logic, y_spt[i])  # compute CE loss
                loss_triplet = self.triplet_loss(embedding, y_spt[i])  # compute triplet loss
                loss = loss_ce + weight_tri * loss_triplet
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                output_q = self.net(x_qry[i], fast_weights, bn_training=True)
                embedding_q, logic_q = output_q[0], output_q[1]
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_ce_q = F.cross_entropy(logic_q, y_qry[i])
                loss_triplet_q = self.triplet_loss(embedding_q, y_qry[i])
                loss_q = loss_ce_q  + weight_tri * loss_triplet_q
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    predict_q = F.softmax(logic_q, dim=1).argmax(dim=1)
                    correct = torch.eq(predict_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
        # end of all tasks

        loss_q = losses_q[-1] / task_num  # sum over all losses on query set across all tasks

        self.meta_optim.zero_grad()  # optimize theta parameters
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        acc = np.array(corrects) / (query_size * task_num)
        return acc

    def fine_tuning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [set_size, c_, h, w]
        :param y_spt:   [set_size]
        :param x_qry:   [query_size, c_, h, w]
        :param y_qry:   [query_size]
        :return:
        """
        assert len(x_spt.shape) == 4

        query_size = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias, we fine_tune on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logic = net(x_spt)
        loss = F.cross_entropy(logic, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [set_size, n_way]
            logic_q = net(x_qry, net.parameters(), bn_training=True)
            # [set_size]
            predict_q = F.softmax(logic_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(predict_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [set_size, n_way]
            logic_q = net(x_qry, fast_weights, bn_training=True)
            # [set_size]
            predict_q = F.softmax(logic_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(predict_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logic = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logic, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logic_q = net(x_qry, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logic_q, y_qry)  # loss_q will be overwritten and just keep the loss_q on last update step.

            with torch.no_grad():
                predict_q = F.softmax(logic_q, dim=1).argmax(dim=1)
                correct = torch.eq(predict_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        acc = np.array(corrects) / query_size

        return acc


def main():
    pass


if __name__ == '__main__':
    main()
