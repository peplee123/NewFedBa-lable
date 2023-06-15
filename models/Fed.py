#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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
    #----------------------------------
    # 通过数据的概率使用gmm去做聚类
    # data = torch.cat([x.unsqueeze(0) for x in client_distributed])
    #
    # # 定义 GMM 模型，聚类数为 3
    # gmm = GaussianMixture(n_components=4)
    #
    # # 训练模型
    # gmm.fit(data)
    #
    # # 预测聚类
    # labels = gmm.predict(data)
    # print("gmm",labels)
    #------------------------------------
    # distances = []
    # for i in range(len(client_distributed)):
    #     distance_i = []
    #     for j in range(len(client_distributed)):
    #         if i != j:
    #             distance = torch.dist(client_distributed[i], client_distributed[j])
    #             distance_i.append(distance)
    #         else:
    #             distance_i.append(0.0)
    #     distances.append(distance_i)
    # print(distances)
    # 这里是通过计算的距离取反比去做聚合更改方式的
    # reciprocal_matrix = np.zeros_like(distances)
    # for i in range(len(distances)):
    #     reciprocal_matrix[i][i] = 0.5*np.sum([torch.reciprocal(x) for x in distances[i] if x != 0.0])
    #     for j in range(len(distances[0])):
    #         if i != j:
    #             reciprocal_matrix[i][j] = torch.reciprocal(distances[i][j])
    # print('-------------------------------')
    # print(reciprocal_matrix)

    # 这里是做的kmeans聚类的
    num_clusters = 2
    num_iterations = 50
    client_distributed = [item for item in client_distributed]
    client_distributed = torch.tensor([item.numpy() for item in client_distributed])
    client_distributed = client_distributed.cpu()
    # print(client_distributed)

    # index_dict = make_cluster_cengci(client_distributed)

    # kmeans = KMeans(n_clusters=num_clusters, max_iter=num_iterations)
    # labels = kmeans.fit_predict(client_distributed.numpy())

    # labels = wasserstein_kmeans(client_distributed.numpy(), n_clusters=num_clusters)

    # draw_distributed(client_distributed)
    labels = cluster_dbscan(client_distributed.numpy(), eps=0.25, min_samples=10)
    # client_distributed = StandardScaler().fit_transform(client_distributed)

    # 执行DBSCAN聚类
    # eps_values = np.linspace(0.1, 2.0, num=20)
    # min_samples = 5
    # num_clusters = []
    #
    # for eps in eps_values:
    #     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    #     labels = dbscan.fit_predict(client_distributed)
    #     num_clusters.append(len(set(labels)) - (1 if -1 in labels else 0))
    #
    # # 绘制肘部图
    # plt.plot(eps_values, num_clusters, marker='o')
    # plt.xlabel('Epsilon')
    # plt.ylabel('Number of Clusters')
    # plt.title('Elbow Curve for DBSCAN')
    # plt.savefig('elbow_curve.png')
    # plt.show()
    #
    #
    # print('labels',labels)

    index_dict = {}
    for i in range(len(labels)):
        if labels[i] not in index_dict:
            index_dict[labels[i]] = []
        index_dict[labels[i]].append(i)
    print('index_dict',index_dict)

    w_global_dict = {}
    # 将 w_locals[0] 转换为 PyTorch 张量
    w_locals_tensor = {}
    for key, value in w_locals[0].items():
        # w_locals_tensor[key] = torch.tensor(value)
        w_locals_tensor[key] = value.clone().detach().requires_grad_(True)

    w_global_avg = FedAvg(w_locals)
    # 创建全 0 张量，并将其赋值给 w_avg
    w_avg = {}
    for key, value in w_locals_tensor.items():
        w_avg[key] = torch.zeros_like(value)
    for j in index_dict.keys():
        for k in w_avg.keys():
            for i in index_dict[j]:
                w_avg[k] += w_locals[i][k]
            w_avg[k] = 0.5 * torch.div(w_avg[k], len(index_dict[j])) + 0.5 * w_global_avg[k]
        w_global_dict[j] = copy.deepcopy(w_avg)

    return w_global_dict, index_dict


def cluster_dbscan(distributed, eps=0.3, min_samples=5):
    # metric：距离度量方法，用于计算样本之间的距离。可以是字符串形式的距离度量名称（如'euclidean'、'manhattan'、'cosine'等）
    dbscan = DBSCAN(eps=eps, min_samples=min_samples,metric='cosine')
    dbscan.fit(distributed)
    labels = dbscan.labels_
    return labels


def draw_distributed(distributed):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=distributed)
    plt.title('Probability Distribution of Each Client')
    plt.xlabel('Client')
    plt.ylabel('Probability')

    plt.savefig("1.jpg")

def kmodes_clustering(data, num_clusters, num_iterations):
    num_samples, num_features = data.size()

    # 随机初始化聚类中心
    centroids = data[torch.randperm(num_samples)[:num_clusters]]

    for _ in range(num_iterations):
        # 计算每个样本与聚类中心之间的距离
        distances = torch.cdist(data, centroids, p=0)

        # 根据距离选择最近的聚类中心
        _, labels = torch.min(distances, dim=1)

        # 更新聚类中心
        for cluster in range(num_clusters):
            cluster_samples = data[labels == cluster]

            # 计算每个特征的众数
            modes = torch.mode(cluster_samples, dim=0).values

            # 更新聚类中心为众数
            centroids[cluster] = modes

    return labels
if __name__ == '__main__':
    cluster_dbscan([[0,0,0,0,0],[2,2,2,2,2],[3,3,3,3,3],[9,9,9,9,9]])