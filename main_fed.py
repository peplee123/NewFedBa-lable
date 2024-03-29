#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import mnist_iid, noniid, cifar_iid,cifar_noniid,build_noniid,bingtai_mnist, draw_data_distribution
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM
from models.Fed import FedAvg, FedBa, NewFedBa, FedCluster, cluster_kmeans,cluster_dbscan
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        print(len(dataset_train))
        print(len(dataset_test))
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            # dict_users = build_noniid(dataset_train, args.num_users, 1)
            train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, 0.1)
            print(len(train_dict_users))
            # draw_data_distribution(train_dict_users, dataset_train, 10)
            # draw_data_distribution(test_dict_users, dataset_train, 10)
            # dict_users = bingtai_mnist(dataset_train, args.num_users,2,random.randint(200, 2500))
            # for client_id, indices in train_dict_users.items():
            #     print(f"Client {client_id}: {indices}")
    elif args.dataset == 'cifar10':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = build_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar100':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR100('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR100('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = build_noniid(dataset_train, args.num_users,10)
    elif args.dataset == 'fashion-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar100':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        # net_glob = torch.load('model80.pt')
        net_glob = CNNMnist(args=args).to(args.device)  # 先定义相同结构的模型对象
        # net_glob.load_state_dict(torch.load('model80.pt', map_location=torch.device('cpu')))
        # net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    # copy weights

    w_glob = net_glob.state_dict()

    learning_rate = [args.lr for i in range(args.num_users)]
    #
    # 用完软预测做完聚类，我们后面是否还可以利用一些软预测来做聚合方式的修改
    all_client_w = [copy.deepcopy(net_glob.state_dict()) for _ in range(args.num_users)]
    # 定义一个聚类表字典
    for iter in range(args.epochs):
        # 每个簇单独做fl

        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        allclient_distributed = []
        print('选取的客户端编号', idxs_users)
        for idx in idxs_users:
            args.lr = learning_rate[idx]
            local = LocalUpdate(
                args=args,
                dataset=dataset_train,
                idxs=train_dict_users[idx],
                dataset_test=dataset_train,
                test_idxs=test_dict_users[idx],
                common_dataset_test=dataset_test,
                w_global=w_glob
            )
            net = copy.deepcopy(net_glob)
            net.load_state_dict(all_client_w[idx])
            w, loss, curLR, distribution = local.train(net)
            learning_rate[idx] = curLR
            w_locals.append(copy.deepcopy(w))
            allclient_distributed += distribution

        gmm = GMM(n_components=2).fit(allclient_distributed)
        cluster_labels = gmm.predict(allclient_distributed)
        print(cluster_labels)
        index_dict = {}
        for i in range(len(cluster_labels)):
            if cluster_labels[i] not in index_dict:
                index_dict[cluster_labels[i]] = []
            index_dict[cluster_labels[i]].append(i)
        print('index_dict', index_dict)
        # {0: [1,5,4], 1:[8]}

        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

        accuracy, test_loss = test_img(net_glob, dataset_test, args)
        print(f"fed avg acc {accuracy} loss {test_loss}")

        for c_id, cluster in index_dict.items():
            cluster_w_list = []
            for client in cluster:
                cluster_w_list.append(w_locals[client])

            cluster_w = FedAvg(cluster_w_list)
            cluster_net = copy.deepcopy(net_glob)
            cluster_net.load_state_dict(cluster_w)
            accuracy, test_loss = test_img(cluster_net, dataset_test, args)
            print(f"cluster {c_id} acc {accuracy} loss {test_loss}")
            for client in cluster:
                all_client_w[client] = cluster_w
