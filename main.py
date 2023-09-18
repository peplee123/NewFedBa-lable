#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import random
from utils.sampling import noniid, build_noniid,build_noniid_agnews, separate_data
from utils.options import args_parser
from utils.dataset import CustomAGNewsDataset
from models.Update import LocalUpdate, DatasetSplit
from models.Nets import LeNet5Cifar,LeNet5Fmnist,resnet18,MLP, CNNMnist, CNNCifar, CNNFemnist, fastText, CNNTinyImage, CNNCifar100,ResNet9
from models.Fed import FedAvg,FedBa, NewFedBa
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare, ImageFolder_custom, CustomImageDataset
from torch.utils.data import ConcatDataset, Dataset
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')




    if args.dataset == 'svhn':

        trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # 下载和加载训练集和测试集
        dataset_train = datasets.SVHN(root='./data/svhn/', split='train', download=True, transform=trans_svhn)
        dataset_test = datasets.SVHN(root='./data/svhn/', split='test', download=True, transform=trans_svhn)

        if args.iid:
            dict_users = svhn_iid(dataset_train, args.num_users)  # 注意：你需要定义一个svhn_iid函数或者复用mnist_iid
        else:
            # 合并训练集和测试集
            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")

    # load dataset and split users
    if args.dataset == 'agnews':
        # 基本的英文tokenizer
        max_len = 200
        tokenizer = get_tokenizer('basic_english')

        # 加载AG News数据集
        trainset, testset = AG_NEWS(root="./data/agnews")

        trainlabel, traintext = list(zip(*trainset))
        testlabel, testtext = list(zip(*testset))

        dataset_text = []
        dataset_label = []

        dataset_text.extend(traintext)
        dataset_text.extend(testtext)
        dataset_label.extend(trainlabel)
        dataset_label.extend(testlabel)

        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(map(tokenizer, iter(dataset_text)), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: int(x) - 1


        def text_transform(text, label, max_len=0):
            label_list, text_list = [], []
            for _text, _label in zip(text, label):
                label_list.append(label_pipeline(_label))
                text_ = text_pipeline(_text)
                padding = [0 for i in range(max_len - len(text_))]
                text_.extend(padding)
                text_list.append(text_[:max_len])
            return label_list, text_list


        label_list, text_list = text_transform(dataset_text, dataset_label, max_len)

        text_lens = [len(text) for text in text_list]
        # max_len = max(text_lens)
        # label_list, text_list = text_transform(dataset_text, dataset_label, max_len)

        text_list = [(text, l) for text, l in zip(text_list, text_lens)]

        text_list = np.array(text_list, dtype=object)
        label_list = np.array(label_list)

        dataset_train = CustomAGNewsDataset(text_list, label_list)
        # 判断是IID还是non-IID的划分
        if args.iid:
            dict_users = mnist_iid(train_dataset, args.num_users)
        else:
            if args.type == 'dir':
                print("dir")
                # X, y = separate_data((text_list, label_list), args.num_users, args.num_classes, niid=args.iid is False, balance=False, partition=args.type, class_per_client=args.bingtai, alpha=args.dir)
                # print(X, y)
                train_dict_users, test_dict_users = build_noniid_agnews([(text_list, label_list)], args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, train_dataset, args.num_users)
            else:
                print("type is none")

    elif args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:

            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
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
            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
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
            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
    elif args.dataset == 'fashion-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:

            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
    elif args.dataset == "tiny-image":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = ImageFolder_custom(root='./data/tiny-imagenet-200/train/', transform=transform)
        testset = ImageFolder_custom(root='./data/tiny-imagenet-200/val/', transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        if args.iid:
            exit('Error: not support')
        else:
            dataset_train = ConcatDataset([CustomImageDataset(dataset_image, dataset_label)])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")

    # img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = LeNet5Cifar(num_classes=10).to(args.device)
        print("model is lenet")
    elif args.model == 'cnn' and args.dataset == 'cifar100':
        net_glob =ResNet9(in_channels=3,num_classes=100).to(args.device)
        print("model is lenet")
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = LeNet5Fmnist(num_classes=10).to(args.device)
        print("model is lenet")
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.dataset == 'agnews':
        net_glob = fastText(hidden_dim=32, vocab_size=98635, num_classes=4).to(args.device)
    elif args.dataset == 'tiny-image':
        net_glob = resnet18(num_classes=200).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'svhn':
        net_glob = LeNet5Cifar(num_classes=10).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # w_glob = net_glob.state_dict()
    w_glob = copy.deepcopy(net_glob)

    # training
    #定义一个客户端的编号表
    num_client = []
    #定义一个聚类表字典
    cluster_count = args.num_users
    cluster_dict = {k: [] for k in range(cluster_count+1)}
    cluster_dict[0] = range(args.num_users)
    cluster_model_list = [copy.deepcopy(net_glob) for _ in range(cluster_count+1)]
    client_model_list = [copy.deepcopy(net_glob) for _ in range(args.num_users)]
    acc_test = []
    acc_client = []
    loss_train = []
    learning_rate = [args.lr for i in range(args.num_users)]
    '''
     # 这里是仿照那个中文论文的框架写的预训练的代码
    local_model_list = [copy.deepcopy(net_glob) for _ in range(args.num_users)]

    allclient_distributed = []
    for idx in train_dict_users:
        print(f"start client {idx}")
        local_args = copy.deepcopy(args)
        local_args.local_ep = 3
        local = LocalUpdate(args=local_args, dataset=dataset_train, idxs=train_dict_users[idx], test_idxs=test_dict_users[idx])
        w, loss, curLR, everyclient_distributed = local.train(net=copy.deepcopy(local_model_list[idx]).to(args.device))
        local_model_list[idx] = copy.deepcopy(net_glob.load_state_dict(w))
        allclient_distributed.append(everyclient_distributed)

    allclient_distributed = np.array([np.array(tensor.tolist()) for sublist in allclient_distributed for tensor in sublist])
    print(allclient_distributed)
    cluster_labels = cluster_dbscan(allclient_distributed)
    print(cluster_labels)
    index_dict = {}
    for i in range(len(cluster_labels)):
        if cluster_labels[i] not in index_dict:
            index_dict[cluster_labels[i]] = []
        index_dict[cluster_labels[i]].append(i)
    print('index_dict', index_dict)
    '''
#用完软预测做完聚类，我们后面是否还可以利用一些软预测来做聚合方式的修改
    user_local_dict = {}
    for i in range(args.num_users):
        user_local_dict[i] = LocalUpdate(args=args, dataset=dataset_train, idxs=train_dict_users[i],
                            test_idxs=test_dict_users[i])

    for iter in range(args.epochs):
        allclient_distributed = []
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print('选取的客户端编号', idxs_users)
        '''
        for key, value in index_dict.items():

            print("Key:", key)
            print("Values:")
            for v in value:
                print(v)
            print("-----")
        '''
        acc_total = 0

        for idx in idxs_users:
            '''
             for cluster_id,cluster_list in cluster_dict.items():
                if idx in cluster_list:
                    c_id = cluster_id
                    break
            # '''
            # [0.0312, 0.0631, 0.0410, 0.0540, 0.3628, 0.0431, 0.0555, 0.0491, 0.0393,
            #  0.2608]
            # [0.0368, 0.0650, 0.0476, 0.1223, 0.2539, 0.0467, 0.0599, 0.0527, 0.0447,
            #  0.2704]
            args.lr = learning_rate[idx]
            local = user_local_dict[idx]
            w, loss, curLR,everyclient_distributed, acc = local.train(net=copy.deepcopy(client_model_list[idx]).to(args.device))
            acc_total += acc
            learning_rate[idx] = curLR
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            allclient_distributed.append(everyclient_distributed)
        acc = acc_total/len(idxs_users)
        print(f"avg acc {acc}")
        acc_client.append(acc)
        # new_allclient_distributed = list(map(lambda x: list(map(float, x[0])), allclient_distributed))
        tensor_list = [item[0] for item in allclient_distributed]
        # print("传入聚类的数据",tensor_list)
        w_global_dict, index_dict = NewFedBa(w_locals, tensor_list)
        # 这里返回的index_dict 代表的id是tensor_list内的索引，而真实的客户端id是idxs_users
        # 也就是说如果要取到真实的客户端id，需要取cid=idxs_users[id]
        # print("通过软标签聚类的全局模型聚合完毕")
        cluster_acc_total = 0
        for k, idx_list in index_dict.items():
            for idx in idx_list:
                c_id = idxs_users[idx]
                # w_global_dict[k] =interpolate_models(w_global_dict[k],w_glob,0.5)
                # w_locals[idx] = copy.deepcopy(w_global_dict[k])
                client_model_list[c_id].load_state_dict(w_global_dict[k])
            cluster_model_list[k].load_state_dict(w_global_dict[k])
            cluster_dict[k] = idx_list

            dataset_test_idx = np.concatenate([test_dict_users[idxs_users[idx]] for idx in idx_list])

            # print accuracy
            acc_t, loss_t = test_img(cluster_model_list[k], DatasetSplit(dataset_train, dataset_test_idx), args)
            cluster_acc_total += acc_t
            # print("Round {:3d},cluster {} Testing accuracy: {:.2f}".format(iter, k, acc_t))
        acc = cluster_acc_total / len(index_dict)
        acc_test.append(acc)
        print(f"Round {iter:3d}, cluster avg acc {acc}")


        # copy weight to net_glob
        # net_glob.load_state_dict(w_glob)
        # net_glob.load_state_dict(w_glob.state_dict())
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        rootpath = './log'
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)
        accfile = open(rootpath + '/acc_cluster_avg_file_fed_{}_{}_{}_iid{}_{}_{}_{}.dat'.
                       format(args.dataset, args.model, args.epochs, args.iid,args.lr,args.local_bs,args.beizhu), "w")
        accfile1 = open(rootpath + '/loss_file_fed_{}_{}_{}_iid{}_{}_{}_{}.dat'.
                       format(args.dataset, args.model, args.epochs, args.iid,args.lr,args.local_bs,args.beizhu), "w")
        accfile2 = open(rootpath + '/acc_client_avg_file_fed_{}_{}_{}_iid{}_{}_{}_{}.dat'.
                        format(args.dataset, args.model, args.epochs, args.iid, args.lr, args.local_bs,args.beizhu), "w")
        for ac in acc_test:
            sac = str(ac)
            accfile.write(sac)
            accfile.write('\n')
        accfile.close()
        for ac in loss_train:
            sac = str(ac)
            accfile1.write(sac)
            accfile1.write('\n')
        accfile1.close()
        for ac in acc_client:
            sac = str(ac)
            accfile2.write(sac)
            accfile2.write('\n')
        accfile1.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_client)), acc_client)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid{}_{}_{}_acc_client_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,args.lr,args.local_bs,args.beizhu))



