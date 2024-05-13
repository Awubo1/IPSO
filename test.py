import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx

# 读取坐标数据
df_coordinates = pd.read_excel('D:\边缘计算\ipso\ipso\jizhan.xlsx')

# 创建一个空的无向图
G = nx.Graph()

# 添加节点和坐标
for idx, row in df_coordinates.iterrows():
    G.add_node(idx, pos=(row['X'], row['Y']))

# 计算每个节点之间的距离，并添加边
for i in range(len(df_coordinates)):
    for j in range(i + 1, len(df_coordinates)):
        xi, yi = df_coordinates.loc[i, ['X', 'Y']]
        xj, yj = df_coordinates.loc[j, ['X', 'Y']]
        distance = math.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
        G.add_edge(i, j, weight=distance)

# 使用 Prim 算法计算最小生成树
T = nx.minimum_spanning_tree(G)

# 绘制图形
plt.figure(figsize=(12, 10))
pos = nx.get_node_attributes(T, 'pos')

# 标出点的序号和坐标
for idx, (x, y) in pos.items():
    if (x, y) in [(3, 0), (13, 1), (8, 7), (16, 13), (2, 11)]:
        plt.plot(x, y, 'gs', markersize=10, label='配备边缘服务器的基站')  # 绿色方框
        plt.text(x, y, f'({x}, {y})', fontsize=8, ha='center', va='bottom')
    else:
        plt.plot(x, y, 'bo', markersize=8, label='基站')  # 蓝色圆圈
        plt.text(x, y, f'({x}, {y})', fontsize=8, ha='center', va='bottom')

# 绘制连线
nx.draw(T, pos, with_labels=False, node_size=100, edge_color='gray', width=0.5)

# 添加图例
plt.legend()

plt.title('边缘环境拓扑图')
plt.axis('on')
plt.grid(False)
plt.savefig('matrix.png')
plt.show()