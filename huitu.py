import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
from matplotlib import rcParams

config = {
            "font.family": 'serif',
            "font.size": 13,
            "font.weight":"bold",
            "mathtext.fontset": 'stix',
            "font.serif": ['SimHei'],#宋体
            'axes.unicode_minus': False # 处理负号
         }
rcParams.update(config)
# 模拟数据
x= [0.1,0.3,0.5,0.7,0.9] # 负载权重系数
indicator1 = [902,644,248,221,189] # 第一个指标，负载标准差
indicator2 = [5.12,6.38,7.12,7.58,8.12] # 第二个指标，总时延

fig, ax1 = plt.subplots()



color1= 'green'
color2 = 'Blue'  # 设置第二个指标的颜色为不同的
ax1.set_xlabel('负载权重')
color = 'black'
ax1.set_ylabel('负载标准差/KB', color=color)
ax1.plot(x, indicator1, color=color2, linestyle='--', label='负载标准差', marker='o')  # 使用圆形表示点
ax1.tick_params(axis='y', labelcolor=color)

# 实例化第二个y轴
ax2 = ax1.twinx()

ax2.set_ylabel('系统时延/s', color=color)
ax2.plot(x, indicator2, color=color1, linestyle='-', label='系统时延', marker='s')  # 使用正方形表示点
ax2.tick_params(axis='y', labelcolor=color)

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center')



fig.tight_layout()  # 使图表布局紧凑
plt.savefig('one.png')
plt.show()

