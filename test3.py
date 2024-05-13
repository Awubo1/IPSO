import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# 读取数据
df = pd.read_excel('D:\边缘计算\ipso\ipso\jizhan.xlsx')

# 提取坐标数据进行聚类
X = df[['X', 'Y']].values

# 应用KMeans算法找到5个聚类中心
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# 获得聚类中心
centers = kmeans.cluster_centers_
# 初始化一个空列表来保存每个聚类中心最近点的索引
# 继续使用之前的聚类结果：centers

# 初始化一个空字典来保存每个聚类中心最近点的坐标
nearest_points_coords = []

# 遍历每个聚类中心
for center in centers:
    # 计算该中心与所有点的欧氏距离
    distances = np.sqrt(np.sum((X - center)**2, axis=1))
    # 找出最近点的索引
    nearest_point_index = np.argmin(distances)
    # 保存最近点的坐标
    nearest_points_coords.append(X[nearest_point_index])

# 打印出最近点的坐标
print("Coordinates of points closest to each cluster center:")
for coord in nearest_points_coords:
    print(coord)
