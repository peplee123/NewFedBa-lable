# coding: utf-8
import re
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
def read_pacfl_data(data_file):
    pattern = "AVG Final Test Acc: (.*)"
    with open(data_file, encoding="utf-8") as f:
        text = f.read()

    acc_list = re.findall(pattern, text)
    print(len(acc_list))
    return [float(acc) for acc in acc_list]

def read_fedavg_data(data_file):
    pattern = "Global Model Test Acc: (\d+\.\d+)"
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
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = {
        r'$v = 10$; Dir(0.1)': 'o',
        r'$v = 7$;  Dir(0.1)': 'o',
        r'$v = 4$;  Dir(0.1)': 'o',
        r'$v = 1$;  Dir(0.1)': '>',
        r'$v = 10$; Pat(20%)': '>',
        r'$v = 7$;  Pat(20%)': '>',
        r'$v = 4$;  Pat(20%)': '>',
    }

    step = 10
    default_marker = 'x'  # 选择一个默认的标记
    for acc_name, acc_list in acc_dict.items():
        x_values = list(range(0, len(acc_list), step))
        y_values = [acc_list[i] for i in range(0, len(acc_list), step)]
        marker = markers.get(acc_name, default_marker)  # 获取标记，如果不存在则使用默认标记
        plt.plot(x_values, y_values, "-" + marker, color=colors[list(acc_dict.keys()).index(acc_name)],
                 label=f"{acc_name}")

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    #plt.legend()
    plt.legend(loc='lower right')

    plt.grid(True)
    plt.ylim(0, 100)  # 设置纵轴范围，起始为10，终止为100

    import os
    current_directory = os.getcwd()
    # 构建要保存到的文件路径
    save_path = os.path.join(current_directory, "pdf11.9", "CIFAR-10;juleibijiao.pdf")
    # plt.title("SVHN(Non-IID label skew(20%))")

    # plt.title("SVHN(Non-IID label skew(20%))")

    plt.title('CIFAR-10')
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    acc_d = {
        # r'$v = 10$; Pat(20%)': read_feduc_data(
        #     "log/data/cf/acc_client_avg_file_fed_cifar10_cnn_400_iidFalse_0.01_16_cluster10avgcfP20.txt")[:400],
        # r'$v = 7$;   Pat(20%)': read_feduc_data(
        #     "log/data/cf/acc_client_avg_file_fed_cifar10_cnn_400_iidFalse_0.01_16_cluster7cfP20M2.txt")[:400],
        # r'$v = 4$;   Pat(20%)': read_feduc_data(
        #     "log/data/cf/acc_client_avg_file_fed_cifar10_cnn_400_iidFalse_0.01_16_cluster4cfP20M2.txt")[:400],
        r'$v = 4$; Dir(0.1)': read_feduc_data(
            "log/data/cf/acc_client_avg_file_fed_cifar10_cnn_400_iidFalse_0.01_16_cluster4cfD0.1M2.txt")[:400],
        r'$v = 7$; Dir(0.1)': read_feduc_data(
            "log/data/cf/acc_client_avg_file_fed_cifar10_cnn_400_iidFalse_0.01_16_cluster7cfD0.1M2.txt")[:400],
        r'$v = 10$; Dir(0.1)': read_feduc_data(
            "log/data/cf/acc_client_avg_file_fed_cifar10_cnn_400_iidFalse_0.01_16_cluster10avgcfD0.1.txt")[:400],
        r'$v = 1$; Dir(0.1)': read_feduc_data(
            "log/data/cf/acc_client_avg_file_fed_cifar10_cnn_400_iidFalse_0.01_16_cfd0.1c1.dat")[:400],
        # "Cluster10;Dir(0.5)": read_feduc_data(
        #     "log/data/fm/acc_client_avg_file_fed_fashion-mnist_cnn_400_iidFalse_0.01_16_cluster10fmD0.5M2.txt")[:400],
        # "Cluster7;Dir(0.5)": read_feduc_data(
        #     "log/data/fm/acc_client_avg_file_fed_fashion-mnist_cnn_400_iidFalse_0.01_16_cluster7fmD0.5M2.txt")[:400],
        # "Cluster4;Dir(0.5)": read_feduc_data(
        #     "log/data/fm/acc_client_avg_file_fed_fashion-mnist_cnn_400_iidFalse_0.01_16_cluster4fmD0.5M2.txt")[:400],


    }
    # plot_acc_lists(acc_d,zoom_area=(350,400,87,97))
    plot_acc_lists(acc_d)
