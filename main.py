# coding=utf-8
import csv
import math
import sys
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

NUM_LOC = 7  # 取值[0-25]

def construct_map(path, num_loc):
    with open(path) as f:
        _map = {i: (float(j[0]), float(j[1]))
                for i, j in enumerate(csv.reader(f)) if i < num_loc}
    return _map


def construct_cuboid_map(l, w, num_in_one_egde):
    num_loc = num_in_one_egde*4
    assert num_loc >= 4 and num_loc % 4 == 0
    num_in_edge = num_loc / 4 - 1
    listl = np.linspace(0, l, num_in_edge+2)
    listw = np.linspace(0, w, num_in_edge+2)
    _map = {}
    index = 0
    for i in listl:
        _map[index] = (i, 0)
        index += 1
        _map[index] = (i, w)
        index += 1
    for i in listw[1:-1]:
        _map[index] = (0, i)
        index += 1
        _map[index] = (l, i)
        index += 1
    return _map


def draw_map(m):
    plt.figure(1)
    plt.scatter([v[0] for k, v in m.items()],
                [v[1] for k, v in m.items()],
                s=75, alpha=0.4, marker="o")
    plt.show()


def dis_two_loc(m, loc1, loc2):
    l1 = m[loc1]
    l2 = m[loc2]
    return math.sqrt((l1[0]-l2[0])**2+(l1[1]-l2[1])**2)


def cal_route_cost(m, route):
    cost = 0
    length = len(route)
    for i in range(length-1):
        cost += dis_two_loc(m, route[i], route[i+1])

    cost += dis_two_loc(m, route[length-1], route[0])
    return cost


def get_one_route(map, num_loc):
    # 记录了每个节点的到达顺序
    # 2-1-4-3-5-0 -->  节点2->节点1->节点4->节点3...
    route = range(0, num_loc, 1)
    random.shuffle(route)
    return route


def route_swap1(route, a, b):
    rt = copy.copy(route)
    temp = rt[a]
    rt[a] = b
    rt[b] = temp
    return rt


def route_swap2(r, a1, b1):
    rt = copy.copy(r)
    a = min(a1, b1)
    b = max(a1, b1)
    temp = rt[a:b+1]
    for i in xrange(a, b+1, 1):
        rt[i] = temp[b-i]
    return rt


def change_route(route):
    # 返回新路径，不改变原路径
    a = random.randint(0, NUM_LOC-1)
    b = a
    while a == b:
        b = random.randint(0, NUM_LOC-1)

    # 随机选择2个节点，交换路径中的这2个节点的顺序。
    # return route_swap1(route, a, b)

    # 随机选择2个节点，将路径中这2个节点间的节点顺序逆转
    return route_swap2(route, a, b)


def plot_list(l):
    plt.figure()
    plt.plot(l)
    plt.show()


class AnnealingState(object):

    def __init__(self, route):
        self.route = route

    def change_state(self):
        # 在原有路径上继续产生新的路径
        return AnnealingState(change_route(self.route))

    def cal_cost(self, route_map):
        cost = cal_route_cost(route_map, self.route)
        return cost


class AnnealingAlgorithm(object):
    # REFERENCE: "Parallel implementations of the statistical cooling algorithm", Emile H.L. Aarts et.al.
    def __init__(self, start_state, route_map,
                 initial_sequence_length=1000,
                 initial_sequence_num=200,
                 accept_rate_initial=0.98,

                 epsilon=0.01,
                 delta=0.01,

                 sub_chain_max_num=20000,
                 sub_mkv_chain_length=2000,
                 smooth_window_length=20,

                 is_parallel=False):

        self.s_state = start_state
        self.route_map = route_map
        self.is_pal = is_parallel
        self.sub_chain_max_num = sub_chain_max_num
        self.initial_sequence_num = initial_sequence_num
        self.sub_mkv_chain_length = sub_mkv_chain_length
        self.cost_accept_history = []
        self.cost_all_history = []
        self.accept = None
        self.smooth_window_length = smooth_window_length

        self.epsilon = epsilon
        self.delta = delta
        self.accept_rate_initial = accept_rate_initial

        self.c_smooth_prev = None
        self.c0_smooth = None
        self.T_prev = None
        self.initial_sequence_length = initial_sequence_length

    def temperature_initialization(self):
        T_list = []
        while not T_list:
            for i in xrange(self.initial_sequence_num):
                cost_decrease_list = []
                cost_increase_list = []
                s = copy.deepcopy(self.s_state)
                cost = s.cal_cost(self.route_map)
                for i in xrange(self.initial_sequence_length):
                    s_new = s.change_state()
                    cost_new = s_new.cal_cost(self.route_map)
                    if cost_new < cost:
                        cost_decrease_list.append(cost_new - cost)
                        s = s_new
                    else:
                        cost_increase_list.append(cost_new - cost)
                m1 = len(cost_decrease_list)
                m2 = len(cost_increase_list)
                c = np.mean(cost_increase_list)
                temp = m2 / (m2 * self.accept_rate_initial - (1 - self.accept_rate_initial) * m1)
                if temp > 0:
                    T = c / np.log(temp)
                    T_list.append(T)

        print T_list
        return max(T_list)

    def algorithm_initialization(self):
        # 初始化温度
        t = self.temperature_initialization()
        return t

    def run(self):

        if self.is_pal:
            pass
        else:
            # 串行退火
            T = self.algorithm_initialization()
            #T = 100
            self.cost_accept_history.append(self.s_state.cal_cost(self.route_map))
            self.cost_all_history.append(self.s_state.cal_cost(self.route_map))

            for i in xrange(self.sub_chain_max_num):
                sub_cost_accept_history = []
                sub_cost_all_history = []

                while len(sub_cost_accept_history) < self.smooth_window_length:
                    sub_cost_accept_history.append(self.cost_accept_history[-1])
                    sub_cost_all_history.append(self.cost_accept_history[-1])

                    for j in xrange(self.sub_mkv_chain_length):
                        state_new = self.s_state.change_state()
                        cost_new = state_new.cal_cost(self.route_map)
                        delta_cost = cost_new - self.cost_accept_history[-1]

                        # 判断新的状态是否接受
                        if delta_cost <= 0:
                            self.accept = True
                        else:
                            if random.random() < math.exp(-delta_cost / T):
                                self.accept = True
                            else:
                                self.accept = False

                        # 根据接受结果做更新
                        self.cost_all_history.append(cost_new)
                        sub_cost_all_history.append(cost_new)
                        if self.accept:
                            self.cost_accept_history.append(cost_new)
                            sub_cost_accept_history.append(cost_new)
                            self.s_state = state_new

                # 更新退火算法参数
                temp_c = sub_cost_accept_history[-self.smooth_window_length:]
                sigma = np.std(temp_c)
                c_smooth = np.mean(temp_c)

                self.T_prev = T
                T = self.T_prev / (1 + math.log(1 + self.delta) * self.T_prev / 3 / (sigma+0.01))

                if self.c0_smooth is None:
                    self.c0_smooth = c_smooth
                    self.c_smooth_prev = c_smooth
                else:
                    delta = (c_smooth - self.c_smooth_prev) / (T - self.T_prev) * T / self.c0_smooth
                    self.c_smooth_prev = c_smooth
                    if np.abs(delta) < self.epsilon and sigma/c_smooth < 0.01:
                        print("delta is %.2f"%delta)
                        print("c_smooth is %.2f"%c_smooth)
                        print("sigma is %.2f"%sigma)
                        plot_list(sub_cost_accept_history)
                        plot_list(sub_cost_all_history)
                        print("T is %.2f"%T)
                        break

            print i
            plot_list(self.cost_accept_history)
            return self.cost_accept_history[-1]


if __name__ == "__main__":
    # m1 = construct_map("map.csv", NUM_LOC)  # 最多25个地点

    num_one_edge = 4
    num_loc = num_one_edge * 4
    m1 = construct_cuboid_map(10, 10, num_one_edge)
    #draw_map(m1)
    self_adaption = True

    if self_adaption:
        state_initial = AnnealingState(get_one_route(m1, num_loc))
        aa = AnnealingAlgorithm(state_initial, m1)
        cost = aa.run()
        print cost
    else:
        #   模拟退火
        T = 1000
        r = 0.7
        cost_history = []
        route_cur = get_one_route(m1)
        cost_cur = cal_route_cost(m1, route_cur)
        cost_history.append(cost_cur)
        for i in xrange(100000):
            # 剪支
            route_temp = change_route(route_cur)
            cost_cur = cal_route_cost(m1, route_cur)
            delta_cost = cost_cur - cost_history[-1]
            if delta_cost < 0:
                cost_history.append(cost_cur)
                route_cur = route_temp
            elif random.random() < math.exp(-delta_cost / T):
                cost_history.append(cost_cur)
                route_cur = route_temp
                T *= r
        plot_list(cost_history)

    print "FINISHED"




