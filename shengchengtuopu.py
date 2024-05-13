import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# 设置随机种子以便结果可复现
random.seed(42)

# 生成15个坐标，x和y范围都在0到18之间
coordinates = [(random.randint(0, 18), random.randint(0, 18)) for _ in range(18)]

# 将坐标转换成DataFrame并保存到Excel文件中
df_coordinates = pd.DataFrame(coordinates, columns=['X', 'Y'])
df_coordinates.to_excel('jizhan.xlsx', index=False)

# 从15个坐标中随机选择4个并保存到另一个Excel文件中
selected_coordinates = df_coordinates.sample(n=4, random_state=42)
selected_coordinates.to_excel('fuwuqi.xlsx', index=False)

# 生成一个15行15列的数据，其中每行数据中只有2到3个数据为1
data = []
for _ in range(18):
    row = [0] * 18
    num_ones = random.randint(2, 3)  # 每行随机2到3个1
    ones_positions = random.sample(range(18), num_ones)
    for pos in ones_positions:
        row[pos] = 1
    data.append(row)

df_data = pd.DataFrame(data)

# 将数据保存到Excel文件中
df_data.to_excel('e.xlsx', index=False)
