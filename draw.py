# coding: utf-8
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    step = 10
    for acc_name, acc_list in acc_dict.items():
        x_values = list(range(0, len(acc_list), step))
        y_values = [acc_list[i] for i in range(0, len(acc_list), step)]

        plt.plot(x_values, y_values, "-o", color=colors[list(acc_dict.keys()).index(acc_name)],
                 label=f"{acc_name}")

    plt.xlabel("# of round")
    plt.ylabel("Acc")
    plt.legend()
    plt.grid(True)

    # # 设定纵坐标刻度
    # max_acc = max(max(l) for l in acc_lists)
    # min_acc = min(min(l) for l in acc_lists)
    # plt.yticks(np.arange(np.floor(min_acc*10)/10, np.ceil(max_acc*10)/10 + 0.1, 0.1))
    plt.show()


if __name__ == '__main__':
    acc_d = {
        "baseline_cifar10P20.txt": read_pacfl_data("./log/baseline_cifar10P20.txt")[:300],
        "cifar10p30.txt": read_feduc_data("./log/cifar10p30.txt")[:300]
    }
    plot_acc_lists(acc_d)

