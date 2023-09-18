#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from scipy.spatial.distance import jensenshannon
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import copy
import torch
import math
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def wasserstein_distance(p, q):
    p = torch.FloatTensor(p)
    q = torch.FloatTensor(q)
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)
    cdf_q = cdf_q.unsqueeze(1)
    cdf_p = cdf_p.unsqueeze(1)
    return torch.nn.functional.pairwise_distance(cdf_p, cdf_q, p=1)[0].item()


def wasserstein_kmeans(data, n_clusters):
    # 将数据转换为PyTorch张量
    data = torch.FloatTensor(data)

    # 初始化K-means算法
    kmeans = KMeans(n_clusters=n_clusters)

    # 将K-means算法的距离度量设置为Wasserstein距离
    kmeans._distance_func = lambda x, y: wasserstein_distance(x, y)

    # 执行K-means聚类
    kmeans.fit(data)

    # 返回聚类结果
    return kmeans.labels_


def FedBa(w_global, w_locals):
    a = []
    for i, w_local in enumerate(w_locals):
        weights_k = weight_flatten(w_local)
        weights_i = weight_flatten(w_global)
        sub = (weights_k - weights_i).view(-1)
        sub = torch.dot(sub, sub).detach().cpu()
        sub = abs(np.arctan(sub))
        sub = math.exp(math.cos(sub))
        # 5
        # sub = abs(math.log(np.arctan(sub)))
        # print("处理后的欧氏距离", sub)
        # elif sub == 0:
        #     pass
        # else:
        # sub = math.log(abs(sub))
        # sub = math.log(sub)
        # # 4
        # if sub < 1:
        #     sub = math.log(abs(np.arctan(sub)))
        #     # sub = abs(math.log(np.arctan(sub)))
        # elif sub == 0:
        #     pass
        # else:
        #     sub = math.log(abs(sub))
        # sub = math.log(sub)
        # if sub!=0:
        #     sub = math.log(abs(np.arctan(sub)))
        # else:
        #     sub = tensor.numpy()
        a.append(sub)
    sum_a = sum(a)
    coef = [b / sum_a for b in a]

    print("得到的聚合权重", coef, sum(coef))

    for param in w_global.parameters():
        param.data.zero_()

    for j, w_local in enumerate(w_locals):
        for param, param_j in zip(w_global.parameters(), w_local.parameters()):
            param.data += coef[j] * param_j
    return w_global.state_dict()


def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)

    return params


def NewFedBa(w_locals, client_distributed):
    # 将 client_distributed 转换为张量
    client_distributed = [item for item in client_distributed]
    client_distributed = torch.tensor(np.array([item.numpy() for item in client_distributed]))
    # client_distributed = torch.tensor([item.numpy() for item in client_distributed])
    client_distributed = client_distributed.cpu()
    # 将 client_distributed 转换为 NumPy 数组
    data = client_distributed.numpy()
    # print('data',data)
    # 计算样本之间的 Jensen-Shannon 距离
    dist_matrix = np.zeros((data.shape[0], data.shape[0]))


    # 计算簇内平方和 (WCSS)
    wcss = []
    cluster_range = range(2, min(10, data.shape[0] + 1))  # Limit the range to a maximum of 10 clusters
    for n_clusters in cluster_range:
        labels = fcluster(::Z, n_clusters, 'maxclust')
        cluster_centers = np.zeros((n_clusters, data.shape[1]))
        for cluster_label in range(1, n_clusters + 1):
            cluster_data = data[labels == cluster_label]
            cluster_centers[cluster_label - 1] = np.mean(cluster_data, axis=0)
        cluster_distances = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            cluster_label = labels[i]
            cluster_center = cluster_centers[cluster_label - 1]
            cluster_distances[i] = np.sum((data[i] - cluster_center) ** 2)
        wcss.append(np.sum(cluster_distances))

    # 使用拐点法计算最优簇数
    diff = np.diff(wcss)
    knee = np.argmax(diff) + 1
    optimal_num_clusters = cluster_range[knee]
    print("Optimal number of clusters:", optimal_num_clusters)

    labels = fcluster(Z, optimal_num_clusters, 'maxclust')
    print("Cluster assignments:", labels)
    index_dict = {}
    for i in range(len(labels)):
        if labels[i] not in index_dict:
            index_dict[labels[i]] = []
        index_dict[labels[i]].append(i)
    # print(index_dict)
    # 创建字典来存储每个簇的全局模型
    w_global_dict = {}

    # w_global_avg = FedAvg(w_locals)

    # 创建全 0 张量，并将其赋值给 w_avg
    w_avg = {}
    # for key, value in w_locals[0].items():
    for j in index_dict.keys():
        # 改 每个簇的模型都得先置0
        for key, value in w_locals[0].items():
            w_avg[key] = torch.zeros_like(value)
        for k in w_avg.keys():
            for i in index_dict[j]:
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.div(w_avg[k], len(index_dict[j]))
        w_global_dict[j] = copy.deepcopy(w_avg)
    # print('多个全局模型字典长度',len(w_global_dict),'全局模型聚合索引',index_dict)
    return w_global_dict, index_dict