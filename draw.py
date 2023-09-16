# coding: utf-8
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
def read_pacfl_data(data_file):
    pattern = "AVG Final Test Acc: (.*)"
    with open(data_file, encoding="utf-8") as f:
        text = f.read()

    acc_list = re.findall(pattern, text)
    return [float(acc) for acc in acc_list]


def read_feduc_data(data_file):
    acc_list = []
    with open(data_file, encoding="utf-8") as f:
        for c in f.readlines():
            acc_list.append(float(c.strip().strip("tensor(").strip(")")))
    return acc_list


def plot_acc_lists(acc_dict):
    # 定义线条样式和颜色，您可以根据需要进行更改
    # 定义线条样式和颜色，您可以根据需要进行更改
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    step = 10
    for acc_name, acc_list in acc_dict.items():
        x_values = list(range(0, len(acc_list), step))
        y_values = [acc_list[i] for i in range(0, len(acc_list), step)]
        # plt.plot(x_values, y_values, color=colors[list(acc_dict.keys()).index(acc_name)],
        #             label=f"{acc_name}")
        plt.plot(x_values, y_values, "-o", color=colors[list(acc_dict.keys()).index(acc_name)],
                 label=f"{acc_name}")

    plt.xlabel("communication rounds")
    plt.ylabel("Acc")
    plt.legend()
    plt.grid(True)
    plt.title('Robust analysis-Fmnist')

    # # 设定纵坐标刻度
    # max_acc = max(max(l) for l in acc_lists)
    # min_acc = min(min(l) for l in acc_lists)
    # plt.yticks(np.arange(np.floor(min_acc*10)/10, np.ceil(max_acc*10)/10 + 0.1, 0.1))
    plt.show()


if __name__ == '__main__':
    acc_d = {
        "Pat20": read_feduc_data("./log/cfP20.txt")[:400],
        "Pat30": read_feduc_data("./log/cfP30.txt")[:400],
        "Dir0.1": read_feduc_data("./log/D0.1cf.txt")[:400]
    }
    plot_acc_lists(acc_d)

# def plot_acc_lists(acc_dict):
#     # 定义线条样式和颜色
#     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     step = 1
#     for acc_name, acc_list in acc_dict.items():
#         x_values = list(range(0, len(acc_list), step))
#         y_values = [acc_list[i] for i in range(0, len(acc_list), step)]
#         plt.plot(x_values, y_values, color=colors[list(acc_dict.keys()).index(acc_name)], label=f"{acc_name}")
#
#     plt.xlabel("communication rounds")
#     plt.ylabel("Acc")
#     plt.legend()
#     plt.grid(True)
#     plt.title('Fmnist-Pat30')
#
#     ax = plt.gca()  # 获取当前坐标轴
#
#     # 创建嵌套的坐标轴，位置和大小由参数决定
#     # axins = inset_axes(ax, width="30%", height="30%", loc="lower right")
#
#
#     # 创建一个新的坐标轴
#     axins = plt.axes([0, 0, 1, 1])
#     # 设置该坐标轴的位置。以下的四个数字分别为[x, y, width, height]
#     ip = InsetPosition(ax, [0.6, 0.3, 0.375, 0.35])
#     axins.set_axes_locator(ip)
#
#
#     # 再次绘制300-400的数据
#     for acc_name, acc_list in acc_dict.items():
#         x_values = list(range(350, 400, step))
#         y_values = [acc_list[i] for i in range(350, 400, step)]
#         axins.plot(x_values, y_values, color=colors[list(acc_dict.keys()).index(acc_name)])
#
#     axins.set_xlim(350, 400)  # 设定子图的x轴范围
#     axins.grid(True)  # 设定子图网格
#
#     # 以下代码用于将300-400的x范围框起来并与子图连接
#     ax.indicate_inset_zoom(axins)
#
#     plt.show()
#
# if __name__ == '__main__':
#     acc_d = {
#         "PACFL": [10.4,12.1]+ read_pacfl_data("./log/pacP30fm.txt")[:400],
#         "OUR": read_feduc_data("./log/fmP30.txt")[:400]
#     }
#     plot_acc_lists(acc_d)