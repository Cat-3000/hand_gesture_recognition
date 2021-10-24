import cv2 as cv
import numpy as np

import matplotlib;matplotlib.use('TkAgg')
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import reduce
import random
from collections import defaultdict
import pandas
# from decimal import Decimal          # 用于处理sigmoid函数报错

'''
    逻辑信息：输入多组数据（x，y）：
    函数结构：linear = k*x + b
             sigmoid = 1/(1+np.exp(linear))
             linear2 = k2*sigmoid + b2
             loss = (1/n)*(y - linear2)**2     # **2=平方,loss就是方差，n是n组数据，此例子只使用了一组数据，n=1
'''


def convert_feed_dictionary_to_grapy(feed_dict):
    need_expand = [n for n in feed_dict]  # 得到最外层节点，并放在列表中
    computing_graph = defaultdict(list)  # 构建一个默认value为list的字典，{key：value}，用来建立一个拓扑图

    while need_expand:
        n = need_expand.pop(0)  # 删除列表第一个元素，并且将删除的元素返回，用来进行处理

        if n in computing_graph:
            continue

        if not n.outputs:  # 建立拓扑结构图不用loss做函数输入，loss没有输出
            continue

        if isinstance(n, Placeholder):  # 如果是需要进行人为赋值的量
            n.value = feed_dict[n]

        for m in n.outputs:
            computing_graph[n].append(m)  # 将所有输出节点放在本节点的字典value里面
            need_expand.append(m)

    return computing_graph


def topologic(graph):
    sorted_node = []

    while graph:
        all_nodes_have_inputs = reduce(lambda a, c: a + c, list(graph.values()))  # 所有有输入的节点,放入列表
        all_nodes_have_outputs = list(graph.keys())  # 所有有输出的节点，放入列表

        all_nodes_only_have_inputs_no_outputs = set(all_nodes_have_inputs) - set(all_nodes_have_outputs)
        all_nodes_only_have_outputs_no_inputs = set(all_nodes_have_outputs) - set(all_nodes_have_inputs)

        for node in all_nodes_only_have_outputs_no_inputs:
            sorted_node.append(node)
            graph.pop(node)

            if not graph:  # graph删除完，但是还有最后一层loss层不在graph键上，所有需要单独在最后加上
                node = random.choice(list(all_nodes_only_have_inputs_no_outputs))
                sorted_node.append(node)

    return sorted_node


class Node:
    def __init__(self, inputs=[], name=None, is_trainable=False):
        self.inputs = inputs
        self.outputs = []
        self.name = name
        self.is_trainable = is_trainable
        self.value = None
        self.gradients = dict()

        for nodee in inputs:
            nodee.outputs.append(self)

    def __repr__(self):
        return 'Node_{}'.format(self.name)

    def forward(self):
        pass

    def backward(self):
        pass


class Placeholder(Node):
    def __init__(self, inputs=[], name=None, is_trainable=False):
        Node.__init__(self, inputs=inputs, name=name, is_trainable=is_trainable)  # 重写父类__init__方法

    def __repr__(self):
        return 'Placeholder_{}'.format(self.name)

    def forward(self, monitor=False):
        if monitor is True:
            print('i am {}, i caiculate by human'.format(self.name))

        else:
            # print('i am {}, i caiculate my value :{} by human'.format(self.name, self.value))
            pass

    def backward(self, monitor=False):
        if monitor is True:
            print('i am {}, i got my gradients by myself'.format(self.name))

        else:
            if self.is_trainable:
                self.gradients[self] = self.outputs[0].gradients[self]  # 把自己的偏导再储存到自己的类里面
            else:
                pass


class Linear(Node):
    def __init__(self, inputs=[], name=None, is_trainable=False):
        Node.__init__(self, inputs=inputs, name=name, is_trainable=is_trainable)  # 重写父类__init__方法

    def __repr__(self):
        return 'Placeholder_{}'.format(self.name)

    def forward(self, monitor=False):
        if monitor is True:
            print('i am {}, i caiculate by myself'.format(self.name))

        else:
            self.value = 0.0
            for _ in range(8):
                self.value += self.inputs[_].value*self.inputs[_ + 8].value
            self.value = self.value - self.inputs[16].value
            # print('i am {}, i caiculate my value :{} by myself'.format(self.name, self.value))

    def backward(self, monitor=False):
        if monitor is True:
            print('i am {}, i got my gradients by myself'.format(self.name))

        else:
            for _ in range(8):
                # 五个x变量实际上不需要求导，因为不使用来进行计算
                # self.gradients[self.inputs[_]] = self.outputs[0].gradients[self]*self.inputs[_ + 5].value
                # print('i am {}, i got gradients[{}]:{} by myself'.format(self.name, self.inputs[_],
                #                                                          self.gradients[self.inputs[_]]))

                self.gradients[self.inputs[_+8]] = self.outputs[0].gradients[self]*self.inputs[_].value
                # print('i am {}, i got gradients[{}]:{} by myself'.format(self.name, self.inputs[_+5],
                #                                                          self.gradients[self.inputs[_+5]]))
            self.gradients[self.inputs[16]] = self.outputs[0].gradients[self]*(-1)
            # print('i am {}, i got gradients[w]:{} by myself'.format(self.name, self.gradients[self.inputs[10]]))


class Sigmoid(Node):
    def __init__(self, inputs=[], name=None, is_trainable=False):
        Node.__init__(self, inputs=inputs, name=name, is_trainable=is_trainable)  # 重写父类__init__方法
        self.one_small_circle = []

    def __repr__(self):
        return 'Placeholder_{}'.format(self.name)

    def _sigmoid(self, x):
        # 参数值inx很大时，np.exp(x)可能会发生溢出
        # 用np.minimum&；np.maximum包装了sigmoid函数，可以克服这个问题，主要是最大值与最小值的设置
        sig = 1 / (1 + np.exp(-x))     # Define sigmoid function
        if sig >= 1:
            sig = 0.999999999999999
        if sig <= 0:
            sig = 0.000000000000001
        return sig

        # if x >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        #     # 这样做可以保证np.exp(x)值始终小于1，避免极大溢出
        #     sig = 1.0 / (1 + np.exp(-x))
        # else:
        #     sig = np.exp(x) / (1 + np.exp(x))
        # if sig <= 0:
        #     sig = 0.000000000001
        # return sig

    def forward(self, monitor=False):
        if monitor is True:
            print('i am {}, i caiculate by myself'.format(self.name))

        else:
            x = self.inputs[0]
            self.value = self._sigmoid(x.value)
            self.one_small_circle.append(self.value)
            # print('i am {}, i caiculate my value :{} by myself'.format(self.name, self.value))

    def backward(self, monitor=False):
        if monitor is True:
            print('i am {}, i got my gradients by myself'.format(self.name))

        else:
            x = self.inputs[0]
            self.gradients[x] = self.outputs[0].gradients[self]*self._sigmoid(x.value) * (1-self._sigmoid(x.value))
            # print('i am {}, i got gradients[linear]:{} by myself'.format(self.name, self.gradients[x]))


class Loss(Node):
    def __init__(self, inputs=[], name=None, is_trainable=False):
        Node.__init__(self, inputs=inputs, name=name, is_trainable=is_trainable)  # 重写父类__init__方法
        self.one_small_circle = []

    def __repr__(self):
        return 'Placeholder_{}'.format(self.name)

    def forward(self, monitor=False):
        if monitor is True:
            print('i am {}, i caiculate by myself'.format(self.name))

        else:
            y = self.inputs[0]    # 计算值
            yhat = self.inputs[1]  # 输入值
            self.value = -(yhat.value*np.log(y.value) + (1 - yhat.value)*np.log(1 - y.value))
            self.one_small_circle.append(self.value)
            # print('i am {}, i caiculate my value :{} by myself'.format(self.name, self.value))

    def backward(self, monitor=False):
        if monitor is True:
            print('i am {}, i got my gradients by myself'.format(self.name))

        else:
            y = self.inputs[0]  # 计算值
            yhat = self.inputs[1]  # 输入值
            self.gradients[y] = -(yhat.value*(1/y.value) - (1-yhat.value)*(1/(1-y.value)))
            # print('i am {}, i got gradients[y]:{} by myself'.format(self.name, self.gradients[y]))


def forward(sorted_nodes, monitor=False):
    if monitor is True:       # 模拟模式
        for node in sorted_nodes:
            node.forward(monitor=True)
    else:
        for node in sorted_nodes:
            node.forward()


def backward(sorted_nodes, monitor=False):
    if monitor is True:       # 模拟模式
        for node in sorted_nodes[::-1]:
            node.backward(monitor=True)
    else:
        for node in sorted_nodes[::-1]:
            node.backward()


def optimize(sorted_nodes, learning_rate=1e-2, monitor=False):
    if monitor is True:  # 模拟模式
        pass
    else:
        for node in sorted_nodes:
            if node.is_trainable:
                node.value += learning_rate*node.gradients[node]*(-1)

                # compare = 'large' if node.gradients[node] > 0 else 'small'
                # print('{} value is to {}, i need update myself'.format(node.name, compare))


def main():
    node_x1 = Placeholder(name='x1')
    node_x2 = Placeholder(name='x2')
    node_x3 = Placeholder(name='x3')
    node_x4 = Placeholder(name='x4')
    node_x5 = Placeholder(name='x5')
    node_x6 = Placeholder(name='x5')
    node_x7 = Placeholder(name='x5')
    node_x8 = Placeholder(name='x5')

    node_k1 = Placeholder(name='k1', is_trainable=True)
    node_k2 = Placeholder(name='k2', is_trainable=True)
    node_k3 = Placeholder(name='k3', is_trainable=True)
    node_k4 = Placeholder(name='k4', is_trainable=True)
    node_k5 = Placeholder(name='k5', is_trainable=True)
    node_k6 = Placeholder(name='k5', is_trainable=True)
    node_k7 = Placeholder(name='k5', is_trainable=True)
    node_k8 = Placeholder(name='k5', is_trainable=True)

    node_w = Placeholder(name='w', is_trainable=True)

    node_linear = Linear(name='linear', inputs=[node_x1, node_x2, node_x3, node_x4, node_x5, node_x6, node_x7, node_x8,
                                                node_k1, node_k2, node_k3, node_k4, node_k5, node_k6,
                                                node_k7, node_k8, node_w])
    node_y = Placeholder(name='y')
    node_sigmoid = Sigmoid(name='sigmoid', inputs=[node_linear])
    node_loss = Loss(name='loss', inputs=[node_sigmoid, node_y])

    # 赋初值，用于排序
    feed_dictionary = {
        node_x1: np.random.normal(),
        node_x2: np.random.normal(),
        node_x3: np.random.normal(),
        node_x4: np.random.normal(),
        node_x5: np.random.normal(),
        node_x6: np.random.normal(),
        node_x7: np.random.normal(),
        node_x8: np.random.normal(),

        node_y: 0.5,

        node_k1: np.random.normal(),
        node_k2: np.random.normal(),
        node_k3: np.random.normal(),
        node_k4: np.random.normal(),
        node_k5: np.random.normal(),
        node_k6: np.random.normal(),
        node_k7: np.random.normal(),
        node_k8: np.random.normal(),

        node_w: np.random.normal()
    }

    # 得到各个节点以及其输出节点的字典
    computing_graph = convert_feed_dictionary_to_grapy(feed_dictionary)
    print(computing_graph)

    # 得到新排序列表
    sorted_nodes = topologic(computing_graph)
    print(sorted_nodes)

    learning_rate = 1e-1
    learning_times = 100000  # 学习轮数
    date_all = 1600          # 训练集数据个数
    date_detect = 360       # 测试集数据个数
    losses = []

    # 导入数据
    hands_elements_all = pandas.read_csv("D:/opencv_photo/project_saved//hands_elements_data_all.csv")
    hands_elements = pandas.read_csv("D:/opencv_photo/project_saved//hands_elements_data.csv")  # 已有.csv文件，按照路径导入
    print(hands_elements.shape)
    print(hands_elements.shape[0])
    print(hands_elements.head())  # .head()方法显示前五行数据
    # print(hands_elements.dtypes)          # 查看数据类型
    print(hands_elements.info())  # 查看是否有缺失值
    # print(hands_elements.describe())      # 查看每一列数据的统计特征：标准差，最大值，最小值等
    # print(hands_elements.corr())   # 查看14列数据亮亮之间的相关性系数，是一个对称矩阵横十四，竖十四，两两对应，1：完全正相关，0：完全不相关，-1：完全负相关

    # 从包含全部数据的表中得到最值
    date_used_by_x1 = hands_elements_all['CRIM']  # 使用date_used_by_x[n]：调取第n个数据，从上往下
    date_used_by_x2 = hands_elements_all['ZN']  # 注意数据残缺，这里选择了两个没有残缺的进行学习
    date_used_by_x3 = hands_elements_all['INDUS']
    date_used_by_x4 = hands_elements_all['CHAS']
    date_used_by_x5 = hands_elements_all['NOX']
    date_used_by_x6 = hands_elements_all['RM']
    date_used_by_x7 = hands_elements_all['AGE']
    date_used_by_x8 = hands_elements_all['DIS']

    # 求取最大值，最小值
    x1_min = date_used_by_x1.min()
    x1_max = date_used_by_x1.max()
    print("x1_max:", x1_max, "x1_min:", x1_min)
    x2_min = date_used_by_x2.min()
    x2_max = date_used_by_x2.max()
    print("x2_max:", x2_max, "x2_min:", x2_min)
    x3_min = date_used_by_x3.min()
    x3_max = date_used_by_x3.max()
    print("x3_max:", x3_max, "x3_min:", x3_min)
    x4_min = date_used_by_x4.min()
    x4_max = date_used_by_x4.max()
    print("x4_max:", x4_max, "x4_min:", x4_min)
    x5_min = date_used_by_x5.min()
    x5_max = date_used_by_x5.max()
    print("x5_max:", x5_max, "x5_min:", x5_min)
    x6_min = date_used_by_x6.min()
    x6_max = date_used_by_x6.max()
    print("x6_max:", x6_max, "x6_min:", x6_min)
    x7_min = date_used_by_x7.min()
    x7_max = date_used_by_x7.max()
    print("x7_max:", x7_max, "x7_min:", x7_min)
    x8_min = date_used_by_x8.min()
    x8_max = date_used_by_x8.max()
    print("x8_max:", x8_max, "x8_min:", x8_min)

    date_used_by_x1 = hands_elements['CRIM']  # 使用date_used_by_x[n]：调取第n个数据，从上往下
    date_used_by_x2 = hands_elements['ZN']  # 注意数据残缺，这里选择了两个没有残缺的进行学习
    date_used_by_x3 = hands_elements['INDUS']
    date_used_by_x4 = hands_elements['CHAS']
    date_used_by_x5 = hands_elements['NOX']
    date_used_by_x6 = hands_elements['RM']
    date_used_by_x7 = hands_elements['AGE']
    date_used_by_x8 = hands_elements['DIS']
    date_used_by_y = hands_elements['ZERO']

    # 开始神经网络运算
    for times in range(learning_times):  # 重复循环数据
        for circle in range(date_all):  # 对全部数据的一次遍历
            # 输入输出数据重新赋值
            node_x1.value = (date_used_by_x1[circle] - x1_min) / (x1_max - x1_min)
            node_x2.value = (date_used_by_x2[circle] - x2_min) / (x2_max - x2_min)
            node_x3.value = (date_used_by_x3[circle] - x3_min) / (x3_max - x3_min)
            node_x4.value = (date_used_by_x4[circle] - x4_min) / (x4_max - x4_min)
            node_x5.value = (date_used_by_x5[circle] - x5_min) / (x5_max - x5_min)
            node_x6.value = (date_used_by_x6[circle] - x6_min) / (x6_max - x6_min)
            node_x7.value = (date_used_by_x7[circle] - x7_min) / (x7_max - x7_min)
            node_x8.value = (date_used_by_x8[circle] - x8_min) / (x8_max - x8_min)

            node_y.value = date_used_by_y[circle]
            # 模拟神经网络的计算过程
            # 前向赋值时：输入层不需要计算，直接取数据填，为了配合输入层

            # 前向赋值
            forward(sorted_nodes, monitor=False)
            # 反向求偏导
            backward(sorted_nodes, monitor=False)
            # 更新可训练参数值
            optimize(sorted_nodes, learning_rate=learning_rate, monitor=False)

        # 获得一次遍历的平均计算结果（sigmoid），平均损失（loss）
        # node_sigmoid.one_small_circle = [reduce(lambda a, c: a + c, node_sigmoid.one_small_circle) / date_all]
        node_loss.one_small_circle = [reduce(lambda a, c: a + c, node_loss.one_small_circle) / date_all]

        # if (times + 1) % 1000 == 0:
        #     print("sigmoid:{}, loss:{}".format(node_sigmoid.one_small_circle[0], node_loss.one_small_circle[0]))
        #     print("k1:{}, k2:{}, k3:{}".format(node_k1.value, node_k2.value, node_k3.value))
        #     print("k4:{}, k5:{}".format(node_k4.value, node_k5.value))

        # 清空，准备接收下一波数据
        # losses.append(node_loss.one_small_circle[0])
        node_sigmoid.one_small_ciecle = []
        node_loss.one_small_circle = []

        counter = 0  # 测试正确计数器
        counter_false = 0    # 测试错误计数器
        for circle_detect in range(date_detect):  # 对全部数据的一次遍历
            circle_detect += date_all
            # 输入输出数据重新赋值
            node_x1.value = (date_used_by_x1[circle_detect] - x1_min) / (x1_max - x1_min)
            node_x2.value = (date_used_by_x2[circle_detect] - x2_min) / (x2_max - x2_min)
            node_x3.value = (date_used_by_x3[circle_detect] - x3_min) / (x3_max - x3_min)
            node_x4.value = (date_used_by_x4[circle_detect] - x4_min) / (x4_max - x4_min)
            node_x5.value = (date_used_by_x5[circle_detect] - x5_min) / (x5_max - x5_min)
            node_x6.value = (date_used_by_x6[circle_detect] - x6_min) / (x6_max - x6_min)
            node_x7.value = (date_used_by_x7[circle_detect] - x7_min) / (x7_max - x7_min)
            node_x8.value = (date_used_by_x8[circle_detect] - x8_min) / (x8_max - x8_min)

            node_y.value = date_used_by_y[circle_detect]
            # 模拟神经网络的计算过程
            # 前向赋值时：输入层不需要计算，直接取数据填，为了配合输入层

            # 前向赋值
            forward(sorted_nodes, monitor=False)

            # 将结果显示出来
            if node_y.value == 1:
                if node_sigmoid.value > 0.95:
                    counter += 1
                # if node_sigmoid.value < 0.95:
                #     counter_false += 1
            else:
                if node_sigmoid.value < 0.05:
                    counter += 1
                if node_sigmoid.value > 0.95:
                    counter_false += 1

            # 清除测试集的数据
            node_sigmoid.one_small_ciecle = []
            node_loss.one_small_circle = []

        # 将结果显示出来
        if (times + 1) % 500 == 0:
            print("times:", times)
            print("rate:{}, rate_false:{}".format(counter/date_detect, counter_false/date_detect))
            print("k1:{}, k2:{}, k3:{}".format(node_k1.value, node_k2.value, node_k3.value))
            print("k4:{}, k5:{}, k6:{}".format(node_k4.value, node_k5.value, node_k6.value))
            print("k7:{}, k8:{}, w:{}".format(node_k7.value, node_k8.value, node_w.value))

        if (counter/date_detect) > 0.95:
            print("times:", times)
            print("rate>0.95:")
            print("k1:{}, k2:{}, k3:{}".format(node_k1.value, node_k2.value, node_k3.value))
            print("k4:{}, k5:{}, k6:{}".format(node_k4.value, node_k5.value, node_k6.value))
            print("k7:{}, k8:{}, w:{}".format(node_k7.value, node_k8.value, node_w.value))
            break

    # 预测：取样本第131个正样本和第200个父样本
    node_x1.value = date_used_by_x1[131]
    node_x2.value = date_used_by_x2[131]
    node_x3.value = date_used_by_x3[131]
    node_x4.value = date_used_by_x4[131]
    node_x5.value = date_used_by_x5[131]
    node_x6.value = date_used_by_x6[131]
    node_x7.value = date_used_by_x7[131]
    node_x8.value = date_used_by_x8[131]
    node_y.value = date_used_by_y[131]
    # 前向赋值
    forward(sorted_nodes, monitor=False)
    print("predicte_result:")
    print("node_y:{}, node_sigmoid:{}, node_loss:{}".format(node_y.value, node_sigmoid.value, node_loss.value))
    print("k1:{}, k2:{}, k3:{}".format(node_k1.value, node_k2.value, node_k3.value))
    print("k4:{}, k5:{}, k6:{}".format(node_k4.value, node_k5.value, node_k6.value))
    print("k7:{}, k8:{}, w:{}".format(node_k7.value, node_k8.value, node_w.value))


if __name__ == "__main__":
    main()
