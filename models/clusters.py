import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch import nn
# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, args.num_classes)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
# 定义计算 Wasserstein 距离的函数
def wasserstein_distance(p, q):
    p = torch.FloatTensor(p)
    q = torch.FloatTensor(q)
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)
    cdf_q = cdf_q.unsqueeze(1)
    cdf_p = cdf_p.unsqueeze(1)
    return torch.nn.functional.pairwise_distance(cdf_p, cdf_q, p=1)[0].item()

# 定义计算两个类簇之间的距离的函数
def cluster_distance(cluster1, cluster2):
    dist = 0.0
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            dist += wasserstein_distance(cluster1[i], cluster2[j])
    return dist / (len(cluster1) * len(cluster2))


def make_cluster_cengci(data, cluster_count=5):
    # 初始化聚类结果为每个数据点为一个类簇
    clusters = [[i] for i in range(len(data))]

    # 不断合并最近的两个类簇，直到只剩下一个类簇为止
    while len(clusters) > cluster_count:
        min_dist = float('inf')
        merge_index = (0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = cluster_distance([data[idx] for idx in clusters[i]], [data[idx] for idx in clusters[j]])
                if dist < min_dist:
                    min_dist = dist
                    merge_index = (i, j)
        # 合并最近的两个类簇
        new_cluster = clusters[merge_index[0]] + clusters[merge_index[1]]
        del clusters[merge_index[1]]
        del clusters[merge_index[0]]
        clusters.append(new_cluster)

    clusters_dict = {k: [] for k in range(cluster_count)}
    for i, cluster in enumerate(clusters):
        clusters_dict[i] = copy.deepcopy(cluster)

    return clusters_dict


if __name__ == '__main__':
    # 生成随机数据
    data = np.random.rand(50, 10)
    print(make_cluster(data))
