import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
            "font.family": 'serif',
            "font.size": 15,
            "font.weight":"bold",
            "mathtext.fontset": 'stix',
            "font.serif": ['SimHei'],#宋体
            'axes.unicode_minus': False # 处理负号
         }
rcParams.update(config)
x = np.arange(1, 6)

# 六种算法的性能数据
np.random.seed(0)
data = np.array([
    [3.27, 5.78, 7.12, 10.03, 12.52],
    [3.45, 5.90, 7.40, 10.12, 12.73],#5.3,2.1,5.1,1,1.7
    [3.51, 5.97, 7.63, 10.57, 13.23],#6.9,3.2,6.7,5.6,5,4
    [3.53, 6.12, 7.78, 10.53, 13.26],#7.4,5.6,8.5,4.8,5.6
    [3.68, 6.18, 7.98, 10.73, 13.48],#12,6.5,9.5,6.6,8.1
    [3.88, 6.52, 8.62, 10.98, 13.92]#15.8,11.4,17.5,8.7,10.1
])

# 颜色和图案填充设置
# colors = ['lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'silver', 'gainsboro']
colors = ['lightgrey', 'lightgreen', 'Beige', 'Cyan', 'gold', 'Pink']
patterns = ['', '\\\\', '///', '---', '...', 'xxx']
task=[500,1000,1500,2000,2500]
# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 6))
Algorithm=['IPSO','GA-BPSO','PSO','AHA','GA','RANDOM']
# 绘制柱状图
bar_width = 0.15
for i in range(6):
    ax.bar(x + i * bar_width, data[i], width=bar_width, color=colors[i], label=Algorithm[i], edgecolor='black', hatch=patterns[i])

# 添加标题和轴标签
ax.set_xlabel('总任务量/个')
ax.set_ylabel('系统时延/s')

# 调整x轴刻度
ax.set_xticks(x + bar_width * 2.5)
ax.set_xticklabels([task[i] for i in range(0, 5)])

# 添加图例
ax.legend( loc='center left', bbox_to_anchor=(0,0.8))
plt.savefig('six.png')
# 显示图表
plt.show()

