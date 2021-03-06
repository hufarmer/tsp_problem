# coding=utf-8
import csv
import math
import sys
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import dispy
import functools
import threading

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
    for i in xrange(length-1):
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
    n_l = len(route)
    a = random.randint(0, n_l-1)
    b = a
    while a == b:
        b = random.randint(0, n_l-1)

    # 随机选择2个节点，交换路径中的这2个节点的顺序。
    # return route_swap1(route, a, b)

    # 随机选择2个节点，将路径中这2个节点间的节点顺序逆转
    return route_swap2(route, a, b)


def plot_list(l):
    plt.figure()
    plt.plot(l)
    plt.show()

# 封装所有操作，dispy调用
class AnnealingState(object):

    def __init__(self, route):
        self.route = route
        self.T = None
        self.c_smooth_prev = None
        self.prev_state_route = None
        self.pos_in_pool = None

    def change_state(self):
        # 在原有路径上继续产生新的路径
        return AnnealingState(change_route(self.route))

    def cal_cost(self, route_map):
        cost = cal_route_cost(route_map, self.route)
        return cost


class AnnealingStatePool(object):

    def __init__(self, pool_size):
        global route_map, num_loc
        self.pool_size = pool_size
        self.pool = []
        self.pool_value = []
        self.max_value_index = -1
        self.min_value_index = -1
        self.__route_map = route_map
        self.__num_loc = num_loc

    def initialization(self):
        for i in xrange(self.pool_size):
            self.pool.append(AnnealingState(get_one_route(self.__route_map, self.__num_loc)))
            self.pool_value.append(self.pool[-1].cal_cost(self.__route_map))

        self.max_value_index = self.pool_value.index(max(self.pool_value))
        self.min_value_index = self.pool_value.index(min(self.pool_value))

    def random_choose(self):
        return random.randint(0, self.pool_size-1)

    def compare_and_replace(self, state):
        cost_temp = state.cal_cost(self.__route_map)
        max_cost_in_pool = self.pool_value[self.max_value_index]
        min_cost_in_pool = self.pool_value[self.min_value_index]

        if cost_temp >= max_cost_in_pool: # worse than all states
            return -1
        elif cost_temp <= min_cost_in_pool:  # no worse than all states
            self.pool[self.max_value_index] = state
            self.pool_value[self.max_value_index] = cost_temp
            self.min_value_index = self.max_value_index
            self.max_value_index = self.pool_value.index(max(self.pool_value))
            return 1
        else:
            self.pool[self.max_value_index] = state
            self.pool_value[self.max_value_index] = cost_temp
            self.max_value_index = self.pool_value.index(max(self.pool_value))
            return 0

    def compare_and_replace1(self, state, original_index):
        cost_temp = state.cal_cost(self.__route_map)
        max_cost_in_pool = self.pool_value[self.max_value_index]
        min_cost_in_pool = self.pool_value[self.min_value_index]

        if cost_temp >= max_cost_in_pool:  # worse than all states
            self.pool[original_index] = state
            self.pool_value[original_index] = cost_temp
            self.max_value_index = original_index
            return -1
        elif cost_temp <= min_cost_in_pool:  # no worse than all states
            self.pool[self.max_value_index] = state
            self.pool_value[self.max_value_index] = cost_temp
            self.min_value_index = self.max_value_index
            self.max_value_index = self.pool_value.index(max(self.pool_value))
            return 1
        else:
            self.pool[self.max_value_index] = state
            self.pool_value[self.max_value_index] = cost_temp
            self.max_value_index = self.pool_value.index(max(self.pool_value))
            return 0

    def compare_and_replace_par(self, state):
        cost_temp = state.cal_cost(self.__route_map)
        max_cost_in_pool = self.pool_value[self.max_value_index]
        min_cost_in_pool = self.pool_value[self.min_value_index]

        if cost_temp >= max_cost_in_pool:  # worse than all states
            return -1
        elif cost_temp <= min_cost_in_pool:  # no worse than all states
            self.pool[self.max_value_index] = state
            self.pool_value[self.max_value_index] = cost_temp
            self.min_value_index = self.max_value_index
            self.max_value_index = self.pool_value.index(max(self.pool_value))
            return 1
        else:
            self.pool[self.max_value_index] = state
            self.pool_value[self.max_value_index] = cost_temp
            self.max_value_index = self.pool_value.index(max(self.pool_value))
            return 0



class AnnealingAlgorithm(object):
    # REFERENCE: "Parallel implementations of the statistical cooling algorithm", Emile H.L. Aarts et.al.
    def __init__(self, state=None,
                 initial_sequence_length=1000, initial_sequence_num=200, accept_rate_initial=0.98,
                 epsilon=0.01, delta=0.01,
                 sub_chain_max_num=20000, sub_mkv_chain_length=2000, smooth_window_length=20):
        self.s_state = state
        self.state_pool = None

        self.initial_sequence_length = initial_sequence_length
        self.initial_sequence_num = initial_sequence_num
        self.accept_rate_initial = accept_rate_initial

        self.epsilon = epsilon
        self.delta = delta

        self.sub_chain_max_num = sub_chain_max_num
        self.sub_mkv_chain_length = sub_mkv_chain_length
        self.smooth_window_length = smooth_window_length

        self.cost_accept_history = []
        self.cost_all_history = []
        self.c_smooth_prev = None
        self.T_prev = None
        self.T = None

    def set_temperature(self, T=None):
        global route_map
        if T is None:
            T_list = []
            while not T_list:
                for i in xrange(self.initial_sequence_num):
                    cost_decrease_list = []
                    cost_increase_list = []
                    s = self.s_state
                    cost = s.cal_cost(route_map)
                    for i in xrange(self.initial_sequence_length):
                        s_new = s.change_state()
                        cost_new = s_new.cal_cost(route_map)
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
            self.T = max(T_list)
        else:
            self.T = T

    def set_state(self, state):
        self.s_state = state

    def set_state_pool(self, state_pool):
        self.state_pool = state_pool

    # 需要起始温度和起始状态
    def run_one_episode(self, state, T, sub_mkv_chain_length, smooth_window_length):
        global route_map, num_loc
        cur_state = state
        cur_cost = state.cal_cost(route_map)
        sub_cost_accept_history = [cur_cost, ]
        sub_cost_all_history = [cur_cost, ]

        while len(sub_cost_accept_history) < smooth_window_length:
            sub_cost_accept_history.append(sub_cost_accept_history[-1])
            sub_cost_all_history.append(sub_cost_accept_history[-1])
            for j in xrange(sub_mkv_chain_length):
                state_new = cur_state.change_state()
                cost_new = state_new.cal_cost(route_map)
                delta_cost = cost_new - cur_cost
                # 判断新的状态是否接受
                if delta_cost <= 0:
                    accept = True
                else:
                    if random.random() < math.exp(-delta_cost / T):
                        accept = True
                    else:
                        accept = False
                # 根据接受结果做更新
                sub_cost_all_history.append(cost_new)
                if accept:
                    sub_cost_accept_history.append(cost_new)
                    cur_state = state_new
                    cur_cost = cost_new

        cost_history_windowed = sub_cost_accept_history[-smooth_window_length:]
        return cur_state, cost_history_windowed, sub_cost_accept_history, sub_cost_all_history

    def update_config(self, cost_history_windowed, T, T_prev, c_smooth_prev, delta, epsilon):
        # 更新退火算法参数: 包括温度，状态
        sigma = np.std(cost_history_windowed)
        c_smooth = np.mean(cost_history_windowed)
        T_new = T / (1 + math.log(1 + delta) * T / 3 / (sigma + 0.01))

        if c_smooth_prev is None:
            c_smooth_prev = c_smooth
            finish_flag = False
        else:
            delta_stop = (c_smooth - c_smooth_prev) / c_smooth
            c_smooth_prev = c_smooth
            if np.abs(delta_stop) < epsilon and sigma / c_smooth < 0.01:
                print("delta is %.2f" % delta_stop)
                print("c_smooth is %.2f" % c_smooth)
                print("sigma is %.2f" % sigma)
                print("T is %.2f" % T)
                finish_flag = True
            else:
                finish_flag = False

        return T_new, finish_flag, c_smooth_prev

    def run(self):
        # 自适应串行退火
        assert self.s_state is not None
        global route_map
        self.set_temperature()
        self.cost_accept_history.append(self.s_state.cal_cost(route_map))
        self.cost_all_history.append(self.s_state.cal_cost(route_map))

        for i in xrange(self.sub_chain_max_num):

            state_new, cost_history_windowed, sub_cost_accept_history, sub_cost_all_history = \
                self.run_one_episode(self.s_state, self.T, self.sub_mkv_chain_length, self.smooth_window_length)

            T_new, finish_flag, c_smooth_prev = \
                self.update_config(cost_history_windowed, self.T, self.T_prev, self.c_smooth_prev, self.delta,
                                      self.epsilon)

            if finish_flag:
                break
            print T_new
            self.T_prev = self.T
            self.T = T_new
            self.c_smooth_prev = c_smooth_prev
            self.s_state = state_new
            self.cost_accept_history.extend(sub_cost_accept_history)
            self.cost_all_history.extend(sub_cost_all_history)

        print i
        plot_list(self.cost_accept_history)
        return self.cost_accept_history[-1]

    def run_parallel(self):
        # 并行退火
        self.state_pool.initialization()
        for i in xrange(1000):

            index = self.state_pool.random_choose()
            state = self.state_pool.pool[index]   #!!!!!!deepcopy???
            # print self.state_pool.pool_value[0]
            print self.state_pool.pool[0].T

            self.set_state(state)
            if state.T == None:
                self.set_temperature()
            else:
                self.T = state.T

            state_new, cost_history_windowed, sub_cost_accept_history, sub_cost_all_history = \
                self.run_one_episode(self.s_state, self.T, self.sub_mkv_chain_length, self.smooth_window_length)

            T_new, finish_flag, c_smooth_prev = \
                self.update_config(cost_history_windowed, self.T, self.T_prev, self.c_smooth_prev, self.delta, self.epsilon)

            if finish_flag:
                break

            self.T_prev = self.T
            self.T = T_new
            self.c_smooth_prev = c_smooth_prev
            state_new.T = T_new
            self.s_state = state_new
            self.cost_accept_history.extend(sub_cost_accept_history)
            self.cost_all_history.extend(sub_cost_all_history)

            flag = self.state_pool.compare_and_replace1(self.s_state, index)


        print i
       # plot_list(self.cost_accept_history)
        return self.cost_accept_history[-1]



def dispy_one_episode(state):
    print "func in"
    import random, math, time, copy
    import numpy as np
    global initial_sequence_length, initial_sequence_num, accept_rate_initial
    global sub_mkv_chain_length, smooth_window_length, delta, epsilon
    global num_loc, route_map

    #===================================================================================
    # 读取状态
    prev_state_route = state.prev_state_route
    pos_in_pool = state.pos_in_pool
    c_smooth_prev = state.c_smooth_prev
    #===================================================================================
    is_manual = False
    # 读取温度
    if state.T is None:
        if is_manual:
            T = 100
        else:
            T_list = []
            while not T_list:
                for i in xrange(initial_sequence_num):
                    cost_decrease_list = []
                    cost_increase_list = []
                    s = state
                    cost = state.cal_cost(route_map)
                    for i in xrange(initial_sequence_length):
                        s_new = s.change_state()
                        cost_new = s_new.cal_cost(route_map)
                        if cost_new < cost:
                            cost_decrease_list.append(cost_new - cost)
                            s = s_new
                        else:
                            cost_increase_list.append(cost_new - cost)
                    m1 = len(cost_decrease_list)
                    m2 = len(cost_increase_list)
                    c = np.mean(cost_increase_list)
                    temp = m2 / (m2 * accept_rate_initial - (1 - accept_rate_initial) * m1)
                    if temp > 0:
                        T = c / np.log(temp)
                        T_list.append(T)
            T = max(T_list)
    else:
        T = state.T


    #===================================================================================
    # 计算一段马氏链
    cur_state = state
    cur_cost = state.cal_cost(route_map)
    sub_cost_accept_history = [cur_cost, ]
    sub_cost_all_history = [cur_cost, ]

    while len(sub_cost_accept_history) < smooth_window_length:
        sub_cost_accept_history.append(sub_cost_accept_history[-1])
        sub_cost_all_history.append(sub_cost_accept_history[-1])
        for j in xrange(sub_mkv_chain_length):
            state_new = cur_state.change_state()
            cost_new = state_new.cal_cost(route_map)
            delta_cost = cost_new - cur_cost
            # 判断新的状态是否接受
            if delta_cost <= 0:
                accept = True
            else:
                if random.random() < math.exp(-delta_cost / T):
                    accept = True
                else:
                    accept = False
            # 根据接受结果做更新
            sub_cost_all_history.append(cost_new)
            if accept:
                sub_cost_accept_history.append(cost_new)
                cur_state = state_new
                cur_cost = cost_new

    cost_history_windowed = sub_cost_accept_history[-smooth_window_length:]

    #更新温度等配置参数
    sigma = np.std(cost_history_windowed)
    c_smooth = np.mean(cost_history_windowed)
    T_new = T / (1 + math.log(1 + delta) * T / 3 / (sigma + 0.01))

    if c_smooth_prev is None:
        c_smooth_prev = c_smooth
        finish_flag = False
    else:
        delta_stop = (c_smooth - c_smooth_prev) / c_smooth
        c_smooth_prev = c_smooth
        if np.abs(delta_stop) < epsilon and sigma / c_smooth < 0.01:
            print("=====stop criterion satisfied=====")
            print("delta is %.2f" % delta_stop)
            print("c_smooth is %.2f" % c_smooth)
            print("sigma is %.2f" % sigma)
            print("T is %.2f" % T)
            print("=====************************=====")
            finish_flag = True
        else:
            finish_flag = False

    # 更新状态
    cur_state.T = T_new
    cur_state.c_smooth_prev = c_smooth_prev
    cur_state.prev_state_route = prev_state_route
    # return cur_state
    return cur_state.route, cur_state.T, cur_state.c_smooth_prev, \
           cur_state.prev_state_route, pos_in_pool, \
           finish_flag, cost_history_windowed[-1]

   #===================================================================================


def dispy_one_episode1(state):
    # 包含 计算状态和温度更新 两部分

    # 计算状态
    global sub_mkv_chain_length, smooth_window_length, delta, epsilon
    global route_map, num_loc
    cur_state = state
    cur_cost = state.cal_cost(route_map)
    sub_cost_accept_history = [cur_cost, ]
    sub_cost_all_history = [cur_cost, ]

    while len(sub_cost_accept_history) < smooth_window_length:
        sub_cost_accept_history.append(sub_cost_accept_history[-1])
        sub_cost_all_history.append(sub_cost_accept_history[-1])
        for j in xrange(sub_mkv_chain_length):
            state_new = cur_state.change_state()
            cost_new = state_new.cal_cost(route_map)
            delta_cost = cost_new - cur_cost
            # 判断新的状态是否接受
            if delta_cost <= 0:
                accept = True
            else:
                if random.random() < math.exp(-delta_cost / T):
                    accept = True
                else:
                    accept = False
            # 根据接受结果做更新
            sub_cost_all_history.append(cost_new)
            if accept:
                sub_cost_accept_history.append(cost_new)
                cur_state = state_new
                cur_cost = cost_new

    cost_history_windowed = sub_cost_accept_history[-smooth_window_length:]
    return cur_state, cost_history_windowed, sub_cost_accept_history, sub_cost_all_history

    # 温度更新


def dispy_setup(n_l, r_m):
    import random, copy
    global initial_sequence_length, initial_sequence_num, accept_rate_initial
    global sub_mkv_chain_length, smooth_window_length, delta, epsilon
    global num_loc, route_map
    num_loc = n_l
    route_map = r_m
    initial_sequence_length = 1000
    initial_sequence_num = 200
    accept_rate_initial = 0.98
    sub_mkv_chain_length = 1000
    smooth_window_length = 20
    delta = 0.01
    epsilon = 0.01
    return 0


def dispy_cleanup():
    global initial_sequence_length, initial_sequence_num, accept_rate_initial
    global sub_mkv_chain_length, smooth_window_length, delta, epsilon
    global num_loc, route_map
    del initial_sequence_length, initial_sequence_num, accept_rate_initial
    del sub_mkv_chain_length, smooth_window_length, delta, epsilon
    del num_loc, route_map


def dispy_callback(job):
    '''
    def __init__(self, route):
        self.route = route
        self.T = None
        self.c_smooth_prev = None
        self.prev_state_route = None
        self.pos_in_pool = None

    in main.py
        state.pos_in_pool = i
        state.prev_state_route = route
    '''
    # print job.stdout
    global state_pool, state_pool_lock, pending_jobs, jobs_cond, cost_accept_history
    if job.status == dispy.DispyJob.Finished  \
        or job.status in (dispy.DispyJob.Terminated,
                          dispy.DispyJob.Cancelled,
                          dispy.DispyJob.Abandoned):
        jobs_cond.acquire()
        if job.id:
            pending_jobs.pop(job.id)
            if len(pending_jobs) <= lower_bound:
                jobs_cond.notify()
        jobs_cond.release()

    if job.status == dispy.DispyJob.Finished:
        # 得到一段马氏链的结果，包含更新后的温度及c_smooth_prev，以及该子链是否收敛的结果
        _cur_route, _cur_T, _cur_c_smooth_prev, _prev_route, \
        _pos_in_pool, _finish_flags, _cost = job()

        _cur_state = AnnealingState(_cur_route)
        _cur_state.T = _cur_T
        _cur_state.c_smooth_prev = _cur_c_smooth_prev
        _cur_state.prev_state_route = _prev_route

        state_pool_lock.acquire()
        cost_accept_history.append(_cost)
        #if state_pool.pool[_pos_in_pool].route == _prev_route:
        if state_pool.pool_value[_pos_in_pool] == cal_route_cost(route_map,_prev_route):
            state_pool.compare_and_replace1(_cur_state, _pos_in_pool)
            state_pool.compare_and_replace(_cur_state)
        else:
            # state_pool.compare_and_replace1(_cur_state, _pos_in_pool)
            state_pool.compare_and_replace(_cur_state)
        state_pool_lock.release()


        # 保存信息
        # print "cost is %f" % _cost
        print "temperature is %f" % _cur_T

        # 根据返回结果继续提交任务
        #if finish_flags:
           # stop_flag = True


# deprecated
def dispy_status_callback(status, node, job):
    if status == dispy.DispyJob.Finished:
        print job.stdout

        # 得到一段马氏链的结果，包含更新后的温度及c_smooth_prev，以及该子链是否收敛的结果
        state, finish_flags, cost = job()

        # 更新状态池
        state_pool.pool[0] = state

        # 保存信息
        cost_accept_history.append(cost)
        #print "cost is %f" % cost
        #print finish_flags
        #print job_num_remained

        # 根据返回结果继续提交任务
        if not finish_flags:
            state = state_pool.pool[1]
            print state.T
            print state.c_smooth_prev
            job = cluster.submit_node(node, state, dispy_job_depends=[state])


if __name__ == "__main__":
    # m1 = construct_map("map.csv", NUM_LOC)
    global route_map, num_loc
    num_one_edge = 2
    num_loc = num_one_edge * 4
    route_map = construct_cuboid_map(10, 10, num_one_edge)

    # draw_map(m1)
    self_adaption = False
    is_parallel =  False
    is_parallel_dispy = True

    if self_adaption:
        state_initial = AnnealingState(get_one_route(route_map, num_loc))
        aa = AnnealingAlgorithm(state_initial)
        cost = aa.run()
        print cost
    elif is_parallel:
        state_pool = AnnealingStatePool(2)
        aa = AnnealingAlgorithm()
        aa.set_state_pool(state_pool)
        cost = aa.run_parallel()
        print cost
    elif is_parallel_dispy:
        state_pool = AnnealingStatePool(2)
        state_pool.initialization()
        cost_accept_history = []

        cluster = dispy.JobCluster(dispy_one_episode,
                                   depends=[AnnealingState, change_route,
                                            route_swap2,
                                            cal_route_cost,
                                            dis_two_loc],
                                   setup=functools.partial(dispy_setup, num_loc, route_map),
                                   cleanup=dispy_cleanup,
                                   # cluster_status=dispy_status_callback)
                                   callback=dispy_callback)

        state_pool_lock = threading.Lock()       # 更新状态池时采用
        jobs_cond = threading.Condition()

        # 分发第一轮
        state_size = state_pool.pool_size
        jobs = []
        for i in xrange(state_size):
            print i
            state_pool_lock.acquire()
            state = copy.deepcopy(state_pool.pool[i])
            route = copy.copy(state_pool.pool[i].route)
            state_pool_lock.release()
            state.pos_in_pool = i
            state.prev_state_route = route
            job = cluster.submit(state)
            jobs.append(job)
            job()

        stop_flag = False
        lower_bound = 2
        upper_bound = 10
        job_num_remained = 15000
        pending_jobs = {}

        # 分发其余
        job_id = -1
        while (not stop_flag) and (job_num_remained > 0):
            #print job_num_remained
            job_id += 1
            job_num_remained -= 1
            # 随机选取状态
            index = random.randint(0, state_size-1)
            state_pool_lock.acquire()
            state = copy.deepcopy(state_pool.pool[index])
            route = copy.copy(state_pool.pool[i].route)
            state_pool_lock.release()
            state.pos_in_pool = index
            state.prev_state_route = route
            job = cluster.submit(state)
            jobs_cond.acquire()
            job.id = job_id
            if job.status == dispy.DispyJob.Created or job.status == dispy.DispyJob.Running:
                pending_jobs[job_id] = job
                if len(pending_jobs) >= upper_bound:
                    while len(pending_jobs) > lower_bound:
                        jobs_cond.wait()
            jobs_cond.release()

        if True:
            cluster.wait()
            cluster.print_status()
            cluster.close()

        plot_list(cost_accept_history)
        print state_pool.pool_value
            # 分析state_pool中结果

print "FINISHED"




