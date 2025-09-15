import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from learner import Learner
from torch import optim
from tqdm import tqdm
from scipy.io import savemat
use_gpu = torch.cuda.is_available()

def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov

def train(net, optimizer, dataloader, device):
    """
    :param net: 网络
    :param optimizer: 优化器
    :param dataloader: 训练集
    :param device: 可用显卡
    :return:
    """
    net.train()
    correct = 0
    for x_spt, y_spt in dataloader:
        if torch.cuda.is_available():
            x_spt = x_spt.to(device)
            y_spt = y_spt.long().to(device)
        optimizer.zero_grad()
        logic = net(x_spt)
        loss = F.cross_entropy(logic, y_spt)
        loss.backward()
        optimizer.step()

        predict = F.softmax(logic, dim=1).argmax(dim=1)
        correct += torch.eq(predict, y_spt).sum().item()

    return correct

def test(net, dataloader, device):
    """
    :param net:  网络
    :param x_qry:   [query_size, c_, h, w]
    :param y_qry:   [query_size]
    :return:
    """
    net.eval()
    correct = 0
    with torch.no_grad():
        for x_qry, y_qry in dataloader:
            if torch.cuda.is_available():
                x_qry = x_qry.to(device)
                y_qry = y_qry.long().to(device)
            logic_q = net(x_qry)
            predict_q = F.softmax(logic_q, dim=1).argmax(dim=1)
            correct += torch.eq(predict_q, y_qry).sum().item()
    return correct

def train_and_test(monte_idx, shot, sampling, net, update_lr, update_step_test, train_data_loader, test_data_loader1, test_data_loader2, device):
    """
    :param x_spt:   [set_size, c_, h, w]
    :param y_spt:   [set_size]
    :return:
    """
    optimizer = optim.Adam(net.parameters(), lr=update_lr)
    corrects_train = [0 for _ in range(update_step_test)]
    corrects_test1 = [0 for _ in range(update_step_test)]
    corrects_test2 = [0 for _ in range(update_step_test)]
    for k in range(0, update_step_test):
        corrects_train[k] = train(net, optimizer, train_data_loader, device)
        corrects_test1[k] = test(net, test_data_loader1, device)
        corrects_test2[k] = test(net, test_data_loader2, device)
        f = open(f'results/maml-st_s(lr0.01)_6way_{shot}shot_{sampling}_18dB/cvnn_{monte_idx}_run1.txt', 'a+')
        f.write(str(100.0 * corrects_test1[k] / len(test_data_loader1.dataset)) + '\n')
        f = open(f'results/maml-st_s(lr0.01)_6way_{shot}shot_{sampling}_18dB/cvnn_{monte_idx}_run2.txt', 'a+')
        f.write(str(100.0 * corrects_test2[k] / len(test_data_loader2.dataset)) + '\n')
        print("monte_idx:", monte_idx,
              "epoch:", k,
              ", train acc:", np.array(corrects_train[k]) / len(train_data_loader.dataset),
              ", test1 acc:", np.array(corrects_test1[k]) / len(test_data_loader1.dataset),
              ", test2 acc:", np.array(corrects_test2[k]) / len(test_data_loader2.dataset))

    return np.array(corrects_test1[1] / len(test_data_loader1.dataset)), np.array(corrects_test2[1] / len(test_data_loader2.dataset))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- data loading
    dataset = 'Wi-Fi'
    n_shot = 10
    n_ways = 6
    train_batch_size = 16
    test_batch_size = 16
    lr = 0.001
    epochs = 10
    n_runs = 20
    n_lsamples = n_ways * n_shot
    n_samples = n_lsamples

    import FSLTask

    cfg = {'shot': n_shot, 'ways': n_ways}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas, query1_data, query1_label, query2_data, query2_label = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot, 6).clone().view(n_runs, n_samples)

    # ---- Base class statistics
    base_means = []
    base_cov = []
    base_features_path = "../feature/base_features.plk"
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)

    # ---- classification for each task
    acc1_list = []
    acc2_list = []
    mean_acc1_list = []
    mean_acc2_list = []
    print('Start classification for %d tasks...' % (n_runs))
    for sampling in tqdm(range(50, 850, 50)):
        for monte_idx in tqdm(range(n_runs)):
            support_data = ndatas[monte_idx][:n_lsamples].numpy()
            support_label = labels[monte_idx][:n_lsamples].numpy()

            # ---- distribution calibration and feature sampling
            sampled_data = []
            sampled_label = []
            num_sampled = int(sampling / n_shot)
            for i in range(n_lsamples):
                mean, cov = distribution_calibration(support_data[i], base_means, base_cov, k=2)
                sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
                sampled_label.extend([support_label[i]] * num_sampled)
            sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
            X_aug = np.concatenate([support_data, sampled_data])
            Y_aug = np.concatenate([support_label, sampled_label])

            # mat_features = {'features': X_aug}
            # mat_labels = {'labels': Y_aug}
            # savemat('../feature/novel1_aug_features.mat', mat_features)
            # savemat('../feature/novel1_aug_labels.mat', mat_labels)

            train_dataset = TensorDataset(torch.Tensor(X_aug), torch.Tensor(Y_aug))
            train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

            test_dataset1 = TensorDataset(torch.Tensor(query1_data), torch.Tensor(query1_label))
            test_data_loader1 = DataLoader(test_dataset1, batch_size=test_batch_size, shuffle=True)

            test_dataset2 = TensorDataset(torch.Tensor(query2_data), torch.Tensor(query2_label))
            test_data_loader2 = DataLoader(test_dataset2, batch_size=test_batch_size, shuffle=True)

            # ---- train classifier
            config = [
                ('linear', [1024, 1152]),
                ('relu', [False]),
                ('linear', [n_ways, 1024])
            ]

            # 逻辑回归（Logistic Regression）
            # model = LogisticRegression()
            # model.fit(X_aug, Y_aug)
            # model.fit(support_data, support_label)
            # acc_1 = model.score(query1_data, query1_label)
            # acc_2 = model.score(query2_data, query2_label)
            # print(f"monte {monte_idx}, acc_1 is {acc_1}, and acc_2 is {acc_2}.")

            # K-近邻（K-Nearest Neighbors, KNN）
            # model = KNeighborsClassifier(n_neighbors=5) # 1 for 1-shot, 3 for 5-shot, 5 for 10-shot
            # model.fit(X_aug, Y_aug)
            # model.fit(support_data, support_label)
            # acc_1 = model.score(query1_data, query1_label)
            # acc_2 = model.score(query2_data, query2_label)
            # print(f"monte {monte_idx}, acc_1 is {acc_1}, and acc_2 is {acc_2}.")

            # 支持向量机（Support Vector Machine, SVM）
            # model = SVC(kernel='linear')
            # model.fit(X_aug, Y_aug)
            # model.fit(support_data, support_label)
            # acc_1 = model.score(query1_data, query1_label)
            # acc_2 = model.score(query2_data, query2_label)
            # print(f"monte {monte_idx}, acc_1 is {acc_1}, and acc_2 is {acc_2}.")

            # 决策树（Decision Tree, DT)
            # model = DecisionTreeClassifier()
            # model.fit(X_aug, Y_aug)
            # model.fit(support_data, support_label)
            # acc_1 = model.score(query1_data, query1_label)
            # acc_2 = model.score(query2_data, query2_label)
            # print(f"monte {monte_idx}, acc_1 is {acc_1}, and acc_2 is {acc_2}.")

            # 随机森林 (Random Forest, RF)
            # model = RandomForestClassifier(n_estimators=100)
            # model.fit(X_aug, Y_aug)
            # model.fit(support_data, support_label)
            # acc_1 = model.score(query1_data, query1_label)
            # acc_2 = model.score(query2_data, query2_label)
            # print(f"monte {monte_idx}, acc_1 is {acc_1}, and acc_2 is {acc_2}.")


            # model = Learner(config)
            # if torch.cuda.is_available():
            #     model = model.to(device)
            #
            # acc_1, acc_2 = train_and_test(monte_idx, n_shot, sampling, model, lr, epochs, train_data_loader, test_data_loader1, test_data_loader2, device)

            acc1_list.append(acc_1)
            acc2_list.append(acc_2)
        mean_acc1 = np.mean(np.array(acc1_list))
        mean_acc2 = np.mean(np.array(acc2_list))
        print(f"sampling {sampling}, mean_acc1 is {mean_acc1}, and mean_acc2 is {mean_acc2}.")
        mean_acc1_list.append(mean_acc1)
        mean_acc2_list.append(mean_acc2)
    print(f"mean_acc1_list is {mean_acc1_list}, and mean_acc2_list is {mean_acc2_list}.")

