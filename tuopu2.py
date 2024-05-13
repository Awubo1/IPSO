import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.patches as mpatches
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# 读取坐标数据
df = pd.read_excel('D:/边缘计算/data_and_code/ipso/jizhan.xlsx')
# 提取X和Y坐标
x_values = df['X']
y_values = df['Y']


# 初始化图
G = nx.Graph()

# 添加节点
for index, row in df.iterrows():
    G.add_node(index + 1, pos=(row['X'], row['Y']), label=index + 1)

# 添加所有可能的边及其权重（距离）
for node1 in G.nodes:
    for node2 in G.nodes:
        if node1 < node2:
            x1, y1 = G.nodes[node1]['pos']
            x2, y2 = G.nodes[node2]['pos']
            distance = euclidean_distance(x1, y1, x2, y2)
            G.add_edge(node1, node2, weight=distance)

# 生成最小生成树
MST = nx.minimum_spanning_tree(G)

# 对于MST中的每个节点，如果边数少于3，则添加额外的边
for node in MST.nodes:
    neighbors = list(MST.neighbors(node))
    if len(neighbors) < 3:
        # 找到非邻居节点中距离最近的节点，添加边
        non_neighbors = [n for n in G.nodes if n not in neighbors and n != node]
        non_neighbors.sort(key=lambda x: euclidean_distance(*G.nodes[node]['pos'], *G.nodes[x]['pos']))
        for non_neighbor in non_neighbors:
            if len(MST[node]) < 3 and len(MST[non_neighbor]) < 3:
                MST.add_edge(node, non_neighbor, weight=euclidean_distance(*G.nodes[node]['pos'], *G.nodes[non_neighbor]['pos']))
                if len(MST[node]) == 3:
                    break
# 绘制网络图
pos = nx.get_node_attributes(G, 'pos')
plt.figure(figsize=(12, 10))
# 绘制节点和边
nx.draw(MST, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=1000, node_color='lightblue', edge_color='gray')
# 特别标记边缘服务器的位置
edge_servers = [(17, 6), (2, 18), (0, 2),(8,7)]
edge_server_nodes = [i + 1 for i, (x, y) in enumerate(df.values) if (x, y) in edge_servers]
nx.draw_networkx_nodes(MST, pos, nodelist=edge_server_nodes, node_color='lightgreen', node_shape='s', node_size=1000)

plt.savefig("matrix1.png")
plt.show()
