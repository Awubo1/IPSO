import numpy as np
import matplotlib.pyplot as plt
import xlrd
import random
import math
import copy
from openpyxl import Workbook

class  AHA():
    def __init__(self,length,dim,pop, max_iter):  #   长度  维度，群体数量，迭代次数
        self.PDNumber=(int(pop)*0.7)#发现者数量
        self.SDNumber=(int(pop)*0.2)#意识到有危险预警者数量
        self.s = random.randint(20, 40)  # 单位bit处理所需cpu圈数
        self.s1 = random.randint(1.5e8, 4e8)  # 给每个任务提供的cpu圈数
        self.ST=0.6#预警值
        self.dim = dim  # 维度#
        self.length = length# #长度
        self.pop = pop  # 数量# 这里是初始化蜂群，20个
        self.max_iter = max_iter  # 迭代次数
        self.population = []  # 父代种群
        self.new_popu = []  # 选择算子操作过后的新种群
        self.fitness=[]    ##种群中所有基因的适应度函数
        self.alpha=0.5
        self.iter_bfit=[]     ##每次迭代中的最小的fitness
        self.pbest=[]
        self.now_iter=0
        ####
        self.edge_num = 0
        # self.C = 300000  # 20Mhz=
        self.several_groups = 0  # 初始化随机
        self.tran_v=10*1024##10MB*8/s=Mbit/s
        self.ing_group = 0  # 第几个任务总数序号
        # self.total_num = 0
        self.bs_local = []
        self.edge_local = []
        self.work_size_total = []
        # self.work_num_total = []
        self.edge_index = []
        self.matrix = []
        self.path = []
        self.dis = []
        self.edge_task = []
        self.tran_time = []
        self.edge_use = []
        ####
        self.total_L = []
        self.total_t = []

    def read(self):
        xlsx1 = xlrd.open_workbook(r"D:\边缘计算\ipso\ipso\jizhan.xlsx")
        bs_ = xlsx1.sheets()[0]

        self.bs_num = bs_.nrows - 1  # 行数
        b = bs_.ncols - 1  ##列数

        for i in range(self.bs_num):
            local = bs_.row_values(i + 1)
            self.bs_local.append((local[b - 1], local[b]))

        xlsx2 = xlrd.open_workbook(r"D:\边缘计算\ipso\ipso\fuwuqi.xlsx")
        edge_ = xlsx2.sheets()[0]

        self.edge_num = edge_.nrows

        for i in range(self.edge_num):
            local = edge_.row_values(i)
            self.edge_local.append((local[0], local[1]))

        for i in range(len(self.edge_local)):
            for j in range(len(self.bs_local)):
                if self.edge_local[i] == self.bs_local[j]:
                    self.edge_index.append(j + 1)
        self.edge_index = np.array(self.edge_index)  # edge_index排序
        self.edge_index = list(np.sort(self.edge_index))

        xlsx3 = xlrd.open_workbook(r"D:\边缘计算\ipso\ipso\1500task200-1000.xlsx")
        date_size = xlsx3.sheets()[0]
        size_ncol = date_size.ncols
        self.several_groups = int(size_ncol / self.bs_num)
        m = 0
        for i in range(self.several_groups):
            groups = []
            for j in range(self.bs_num):
                a = date_size.col_values(m)
                del a[0]
                a = list(filter(lambda x: x != '', a))
                m += 1
                groups.append(a)
            self.work_size_total.append(groups)

        xlsx4 = xlrd.open_workbook(r"D:\边缘计算\ipso\ipso\e.xlsx")
        graph_matrix = xlsx4.sheets()[0]
        matrix_ncol = graph_matrix.ncols
        for i in range(matrix_ncol):
            a = graph_matrix.row_values(i)
            self.matrix.append(a)

    def Dijkstra(self,network, s, d):  # 迪杰斯特拉算法算s-d的最短路径，并返回该路径和值
        path = []  # 用来存储s-d的最短路径
        n = len(network)  # 邻接矩阵维度，即节点个数
        fmax = float('inf')
        w = [[0 for _ in range(n)] for j in range(n)]  # 邻接矩阵转化成维度矩阵，即0→max

        book = [0 for _ in range(n)]  # 是否已经是最小的标记列表
        dis = [fmax for i in range(n)]  # s到其他节点的最小距离
        book[s - 1] = 1  # 节点编号从1开始，列表序号从0开始   ##从s点开始便历
        midpath = [-1 for i in range(n)]  # 上一跳列表
        for i in range(n):
            for j in range(n):
                if network[i][j] != 0:
                    w[i][j] = network[i][j]  # 0→max
                else:
                    w[i][j] = fmax
                if i == s - 1 and network[i][j] != 0:  # 直连的节点最小距离就是network[i][j]
                    dis[j] = network[i][j]
        for i in range(n - 1):  # n-1次遍历，除了s节点
            min = fmax
            u = 0
            for j in range(n):
                if book[j] == 0 and dis[j] < min:  # 如果未遍历且距离最小
                    min = dis[j]
                    u = j
            book[u] = 1
            for v in range(n):  # u直连的节 点遍历一遍
                if v == s - 1:
                    dis[v] = 0
                elif dis[v] > dis[u] + w[u][v]:
                    dis[v] = dis[u] + w[u][v]
                    midpath[v] = u + 1  # 上一跳更新
        j = d - 1  # j是序号
        path.append(d)  # 因为存储的是上一跳，所以先加入目的节点d，最后倒置
        while (midpath[j] != -1):
            path.append(midpath[j])
            j = midpath[j] - 1
        path.append(s)
        path.reverse()  # 倒置列表
        a = dis[d - 1]  # 下标要减一
        return path

    def init_Population(self):  # 初始化种群
        ###群数 长度  维度
        self.population = np.zeros((self.pop, self.length,self.dim),dtype= np.int)
        self.choice1 = [i for i in range(self.dim + 1)]
        for i in range(self.pop):
            j=0
            while j<self.length:
                for k in range(self.dim):
                    self.population[i][j][k] = random.choice(self.choice1)
                if np.all(self.population[i][j]==0)==True:
                    continue
                else:
                    j+=1


    def init_task(self):
        self.ing_task=[]
        for i in range(len(self.work_size_total[self.ing_group])):
            task=[]
            for j in range(len(self.work_size_total[self.ing_group][i])):
                task.append(self.work_size_total[self.ing_group][i][j])
            self.ing_task.append(task)  ##任务数量大小确定

    def max_min(self,list):
        min = list[0]
        max = list[0]
        for x in list:
            if x==float('inf'):
                continue
            else:
                if x < min:
                    min = x
                if x > max:
                    max = x
        return max, min


    def com_fit(self):
        self.fitness = []
        self.pop_l = []
        self.pop_t = []
        self.pbest = []
        self.gbest_fitness = []
        self.iter_bfit = []
        self.gbest = []
        self.g_fitness_index = []
        for i in range(self.pop):
            position = self.population[i]
            a = self.com_parallel(position)
            self.pop_l.append(a[0])  # [i][0] = f1
            self.pop_t.append(a[1])
        tmax, tmin = self.max_min(self.pop_t)
        self.tmax = tmax
        self.tmin=tmin
        print(self.tmax)
        lmax, lmin = self.max_min(self.pop_l)
        self.lmax = lmax
        self.lmin=lmin
        print(self.lmax)
        for i in range(self.pop):
            if self.pop_l[i] == float('inf') or self.pop_t[i] == float('inf'):
                self.fitness.append(float('inf'))
            else:
                gfit = self.alpha * (self.pop_l[i] / self.lmin) + (1 - self.alpha) * (self.pop_t[i] / self.tmin)
                # fit=self.alpha*(self.pop_l[i]/self.lmax)+(1-self.alpha)*(self.pop_t[i]/self.tmax)
                self.fitness.append(gfit)
                ##第一次的个体历史最优及基因
                self.pbest.append((gfit, self.population[i]))
        # 得到各个个体的适应度值
        a = np.array(self.fitness)
        b = np.argsort(a)  ###从小到大的坐标序号
        self.gbest_fitness = self.fitness[b[0]]  # 种群中最佳适应度
        self.iter_bfit.append(self.fitness[b[0]])  #len:迭代次数
        self.g_fitness_index = b[0]  ##种群中最佳适应度在坐标中序号
        self.gbest = self.population[b[0]]  # 最佳适应度在整个整体的最优位置

    def com_fit1(self):
        self.fitness = []
        self.pop_l = []
        self.pop_t = []
        for i in range(self.pop):
            position = self.population[i]
            a = self.com_parallel(position)
            self.pop_l.append(a[0])  # [i][0] = f1
            self.pop_t.append(a[1])
        for i in range(self.pop):
            if self.pop_l[i] == float('inf') or self.pop_t[i] == float('inf'):
                self.fitness.append(float('inf'))
            else:
                gfit = self.alpha * (self.pop_l[i] / self.lmin) + (1 - self.alpha) * (self.pop_t[i] / self.tmin)
                # fit=self.alpha*(self.pop_l[i]/self.lmax)+(1-self.alpha)*(self.pop_t[i]/self.tmax)
                self.fitness.append(gfit)
                ##第一次的个体历史最优及基因
                self.pbest.append((gfit, self.population[i]))
            # 得到各个个体的适应度值
            #####更新粒子个体历史最优
            if self.pbest[i][0] > gfit:
                self.pbest[i] = (gfit, self.population[i])
        a = np.array(self.fitness)
        b = np.argsort(a)  ###从小到大的坐标序号
        ##当前种群最优
        min_fitness = self.fitness[b[0]]
        self.iter_bfit.append(min_fitness)  ###len:迭代次数
        ##更新种群历史最优
        if min_fitness < self.gbest_fitness:
            self.gbest_fitness = min_fitness
            self.gbest = self.population[b[0]]  # 初始化整个整体的最优粒子

    def com_parallel(self, gene):
        if_break = False
        self.init_task()
        self.init_task_infor = []
        self.init_task_distance_infor = []  # 保存链路距离大小
        for i in range(self.bs_num):
            AP_corresponding = gene[i]
            task_part = 0
            nonull_index = []
            for z in range(self.dim):
                if AP_corresponding[z] != 0:
                    task_part += 1
                    nonull_index.append(z)
            if task_part == 0:
                print('gene error!')
                if_break = True
                break
            part_numi = int(len(self.ing_task[i]) / task_part)
            task_up = 0
            part_num = []
            for r in range(task_part):
                if r != task_part - 1:
                    part_num.append(task_up + part_numi)
                    task_up += part_numi
                else:
                    part_num.append(len(self.ing_task[i]))
            new_task_dist = []
            for e in range(len(nonull_index)):
                new_task_dist.append(AP_corresponding[nonull_index[e]])
            s = np.array(self.ing_task[i])
            q = s.argsort()
            task_infor = []
            distance_infor = []
            a = 0
            dis = 0
            for j in range(len(self.ing_task[i])):
                if j < part_num[a]:
                    path = self.Dijkstra(self.matrix, i + 1, self.edge_index[nonull_index[a] - 1])
                    distance = 0
                    if (len(path)) > 1:
                        for m in range(len(path) - 1):
                            distance += (np.sqrt(
                                ((self.bs_local[path[m + 1] - 1][0]) - (self.bs_local[path[m] - 1][0])) ** 2 + (
                                        (self.bs_local[path[m + 1] - 1][1]) - (
                                    self.bs_local[path[m] - 1][1])) ** 2))
                    taskj_infor = (self.ing_task[i][j], path[1:])  # (1, [2, 32, 5, 4, 5])
                    distancej_infor = (self.ing_task[i][j], path, distance)
                    distance_infor.append(distancej_infor)
                    task_infor.append(taskj_infor)
                    if j + 1 == part_num[a]:
                        a += 1
            self.init_task_infor.append(task_infor)
            self.init_task_distance_infor.append(distance_infor)
        self.edge_task = [[] for _ in range(self.edge_num)]
        total_ap_tran = 0
        remnant = [0 for _ in range(self.bs_num)]
        null_num = 0
        # 数组模拟队列
        while null_num < self.bs_num:
            if if_break:
                break

            all_ap_full = [False for _ in range(self.bs_num)]
            all_ap_end_task = [0 for _ in range(self.bs_num)]

            for i in range(self.bs_num):
                if self.init_task_infor[i] != []:
                    sum_task_ing = remnant[i]
                    for j in range(len(self.init_task_infor[i])):  ###
                        k = 0
                        if remnant[i] != 0 and j == 0:
                            sum_task_ing += self.init_task_infor[i][j][0] - remnant[i]
                        else:
                            sum_task_ing += self.init_task_infor[i][j][0]
                        # print('sum_task_ing',sum_task_ing)
                        if sum_task_ing < self.tran_v:
                            if j < len(self.init_task_infor[i]) - 1:
                                if sum_task_ing + self.init_task_infor[i][j + 1][0] > self.tran_v:
                                    save = self.tran_v
                                    is_nofull = False
                                    k = j
                                    break
                                elif sum_task_ing + self.init_task_infor[i][j + 1][0] <= self.tran_v:
                                    if j + 1 == len(self.init_task_infor[i]) - 1:
                                        is_nofull = True
                                        k = j + 1
                                        save = sum_task_ing + self.init_task_infor[i][j + 1][0]
                                        break
                                    else:
                                        continue
                            else:
                                k = j
                                save = sum_task_ing
                                is_nofull = True
                                break
                        else:
                            is_nofull = True
                            save = self.tran_v
                    for b in range(k + 1):
                        aim_index = self.init_task_infor[i][0][1][0]
                        task_infor = self.init_task_infor[i][0]
                        task_next_index = task_infor[1][0]
                        task_infor = (task_infor[0], task_infor[1][1:])
                        self.init_task_infor[i].remove(self.init_task_infor[i][0])
                        ##判断
                        if task_infor[1] != []:
                            self.init_task_infor[aim_index - 1].append(task_infor)
                        else:
                            for p in range(len(self.edge_index)):
                                if task_next_index == self.edge_index[p]:
                                    next_index = p
                                    break
                            self.edge_task[next_index].append(task_infor[0])
                    remnant[i] = self.tran_v - save
                all_ap_full[i] = is_nofull
                all_ap_end_task[i] = save
            if False in all_ap_full:
                total_ap_tran += 1
            else:
                short_t = []
                for i in range(self.bs_num):
                    short_t.append(all_ap_end_task[i] / self.tran_v)
                total_ap_tran += max(short_t)
            null_num = 0
            for i in range(self.bs_num):
                if self.init_task_infor[i] == []:
                    null_num += 1
                    continue
        if if_break:
            f1 = float('inf')
            f2 = float('inf')
        else:
            f2 = total_ap_tran
            resource = 0
            for i in range(self.edge_num):
                resource += sum(self.edge_task[i])
            resource_ave = resource / self.edge_num
            resource = []
            self.exec = [[] for _ in range(self.edge_num)]
            max1 = 0
            min1 = 0
            for i in range(self.edge_num):
                for j in range(len(self.edge_task[i])):
                    t_server = (self.edge_task[i][j] * 1024 * 8) / ((self.s1) / self.s)
                    if t_server > max1:
                        max1 = t_server
                    elif t_server < min1:
                        min1 = t_server
                    self.exec[i].append(t_server)
            for i in range(self.edge_num):
                resource_i = (sum(self.edge_task[i]) - resource_ave) ** 2
                resource.append(resource_i)
            L_balance = math.sqrt((sum(resource)) / self.edge_num)
            f1 = L_balance
            f2 = f2 + (max1 + min1) / 2
            print('负载均衡标准差,系统时延:', f1, f2)
            t_server = 0
        return [f1, f2]

        # 模拟蜂鸟的飞行行为来更新位置

    # if mode == 'territorial':upadte1
    #     # 领地觅食：在当前位置附近小范围内搜索
    #     move_direction = np.random.uniform(-1, 1, size=hummingbird.position.shape) * step_size
    # elif mode == 'migratory':upadte3
    #     # 迁徙觅食：向全局最佳位置大幅度移动
    #     move_direction = (best_position - hummingbird.position) * np.random.uniform(0.5, 1.5)
    # elif mode == 'diagonal':
    #     # 对角线觅食：对角线移动探索新区域upadte2
    #     direction = np.random.choice([-1, 1], size=hummingbird.position.shape)
    #     move_direction = direction * step_size * np.sqrt(hummingbird.position.shape[0])
    # else:  # mode == 'guided'
    #     # 引导觅食：根据最佳位置引导搜索方向upadte4
    #     move_direction = (best_position - hummingbird.position) * np.random.uniform(0, 1,
    #                                                                                 size=hummingbird.position.shape)
    def update1(self):#领地觅食：在当前位置附近小范围内搜索
        self.new_popu = np.zeros((self.pop, self.length, self.dim), dtype=np.int)
        self.new_popu1 = np.zeros((self.pop, self.length, self.dim), dtype=np.int)
        step_size=1
        a = np.array(self.fitness)
        ind = np.argsort(a)
        for i in range(self.pop):
            self.new_popu1[i] = self.population[i]
        for i in range(self.pop):
            self.new_popu[i] = self.population[ind[i]]
        for p in range(0,self.pop):
            for i in range(0, self.length):
                for j in range(0, self.dim):
                    self.new_popu1[p][i][j]=random.choice([-1, 1])*step_size+self.new_popu1[p][i][j]
                    if self.new_popu1[p][i][j] > self.choice1[-1]:
                        self.new_popu1[p][i][j] = self.new_popu1[p][i][j] - self.choice1[-1]
                    elif self.new_popu1[p][i][j] < self.choice1[0]:
                        self.new_popu1[p][i][j]=-self.new_popu1[p][i][j]
        for i in range(self.pop):
            self.population[i]=self.new_popu1[i]



    def update2(self):#对角线觅食：对角线移动探索新区域
        self.new_popu = np.zeros((self.pop, self.length, self.dim), dtype=np.int)
        self.new_popu1 = np.zeros((self.pop, self.length, self.dim), dtype=np.int)
        step_size=1
        a = np.array(self.fitness)
        ind = np.argsort(a)
        for i in range(self.pop):
            self.new_popu1[i] = self.population[i]
        for i in range(self.pop):
            self.new_popu[i] = self.population[ind[i]]
        for p in range(0,self.pop):
            for i in range(0, self.length):
                for j in range(0, self.dim):
                    self.new_popu1[p][i][j]=random.choice([-1, 1])*step_size*np.sqrt(self.length)+self.new_popu1[p][i][j]
                    if self.new_popu1[p][i][j] > self.choice1[-1]:
                        self.new_popu1[p][i][j] = self.new_popu1[p][i][j] - self.choice1[-1]
                    elif self.new_popu1[p][i][j] < self.choice1[0]:
                        self.new_popu1[p][i][j]=-self.new_popu1[p][i][j]
        for i in range(self.pop):
            self.population[i]=self.new_popu1[i]

    #move_direction = (best_position - hummingbird.position) * np.random.uniform(0.5, 1.5)
    def update3(self):## 迁徙觅食：向全局最佳位置大幅度移动
        self.new_popu = np.zeros((self.pop, self.length, self.dim), dtype=np.int)
        self.new_popu1 = np.zeros((self.pop, self.length, self.dim), dtype=np.int)
        step_size=1
        a = np.array(self.fitness)
        ind = np.argsort(a)
        for i in range(self.pop):
            self.new_popu1[i] = self.population[i]
        for i in range(self.pop):
            self.new_popu[i] = self.population[ind[i]]
        for p in range(0,self.pop):
            for i in range(0, self.length):
                for j in range(0, self.dim):
                    self.new_popu1[p][i][j]= (self.new_popu[0][i][j]-self.new_popu1[p][i][j])*np.random.uniform(0.5, 1.5)+self.new_popu1[0][i][j]
                    if self.new_popu1[p][i][j] > self.choice1[-1]:
                        self.new_popu1[p][i][j] = self.new_popu1[p][i][j] - self.choice1[-1]
                    elif self.new_popu1[p][i][j] < self.choice1[0]:
                        self.new_popu1[p][i][j]=-self.new_popu1[p][i][j]
        for i in range(self.pop):
            self.population[i]=self.new_popu1[i]

        #     move_direction = (best_position - hummingbird.position) * np.random.uniform(0, 1,
        #                                                                                 size=hummingbird.position.shape)

    def update4(self):  ## 迁徙觅食：向全局最佳位置大幅度移动
        self.new_popu = np.zeros((self.pop, self.length, self.dim), dtype=np.int)
        self.new_popu1 = np.zeros((self.pop, self.length, self.dim), dtype=np.int)
        step_size = 1
        a = np.array(self.fitness)
        ind = np.argsort(a)  ###从小到大的坐标序号
        for i in range(self.pop):
            self.new_popu1[i] = self.population[i]
        for i in range(self.pop):
            self.new_popu[i] = self.population[ind[i]]
        for p in range(0, self.pop):
            for i in range(0, self.length):
                for j in range(0, self.dim):
                    self.new_popu1[p][i][j] = (self.new_popu[0][i][j] - self.new_popu1[p][i][j]) * np.random.uniform(
                        0.9, 1) + self.new_popu1[0][i][j]
                    if self.new_popu1[p][i][j] > self.choice1[-1]:
                        self.new_popu1[p][i][j] = self.new_popu1[p][i][j] - self.choice1[-1]
                    elif self.new_popu1[p][i][j] < self.choice1[0]:
                        self.new_popu1[p][i][j] = -self.new_popu1[p][i][j]
        for i in range(self.pop):
            self.population[i] = self.new_popu1[i]




    def save(self):
        a = np.array(self.fitness)
        b = np.argsort(a)
        self.total_L.append(self.pop_l[b[0]])
        self.total_t.append(self.pop_t[b[0]])
        print(' self.iter_bfit:', self.iter_bfit)
        with open('array.txt', 'w') as f:
            for item in self.iter_bfit:
                f.write("%s\n" % item)
        print('self.total_L:', self.total_L)
        print('self.total_t:', self.total_t)



    def run(self):
        self.read()
        for j in range(self.several_groups):
            self.init_Population()
            self.iter_bfit = []
            if j == 1:
                for k in range(self.max_iter):
                    self.now_iter = k
                    print('finished :,k:', self.ing_group, k)
                    self.init_task()
                    if k == 0:
                        self.com_fit()
                        self.update1()
                        # print(self.lmax,self.tmax)
                    else:
                        # print(self.lmax, self.tmax)
                        if k<0.05*(self.max_iter):
                            self.com_fit1()
                            self.update1()
                        elif k<0.1*(self.max_iter):
                            self.com_fit1()
                            self.update2()
                        elif k < 0.15 * (self.max_iter):
                            self.com_fit1()
                            self.update3()
                        else:
                            self.com_fit1()
                            self.update4()

                self.save()
            print('finished :', self.ing_group)
            self.ing_group += 1



if __name__ == '__main__':
    A=AHA(18,5,120,100)
    A.run()
