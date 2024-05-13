import pandas as pd
import matplotlib.pyplot as plt

# 读取坐标数据
df_coordinates = pd.read_excel('D:\边缘计算\data_and_code\code\jizhan.xlsx')

# 提取X和Y坐标
x_values = df_coordinates['X']
y_values = df_coordinates['Y']

# 绘制散点图
plt.figure(figsize=(10, 8))
plt.scatter(x_values, y_values, color='blue', marker='o')

# 在每个点旁边添加坐标文本
for i in range(len(x_values)):
    plt.text(x_values[i]+0.3, y_values[i], f'({x_values[i]}, {y_values[i]})', fontsize=9)

plt.xlabel('X')
plt.ylabel('Y')
plt.grid(False)  # 去除网格线
plt.xticks(range(0,20))  # X轴刻度
plt.yticks(range(0,20))  # Y轴刻度
plt.xlim(-1, 19)  # 设置X轴的范围
plt.ylim(-1, 19)  # 设置Y轴的范围
plt.show()
