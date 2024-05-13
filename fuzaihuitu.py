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
# 假设有6种算法和5个评估点
x = np.arange(1, 6)

# 六种算法的性能模拟数据
np.random.seed(0)
data = np.array([
    [66, 168, 248, 231, 253],
    [70, 204, 287, 269, 323],##5.6%,17.7%,21.8%,14.2%,21.7%，
    [74, 232, 356, 335, 373],#10.9,27.6,30.4,31.1,32.2
    [138, 473, 621, 700, 823],#52，2,64.5,61,67,69.3,
    [267, 512, 683, 842, 1060],#75.3,67.2,63.7,72.6,76.2
    [871, 1308, 1908, 2645, 3108]#92.5,87.2,84.6,91.3,91.9
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
ax.set_ylabel('负载标准差/KB')

# 调整x轴刻度
ax.set_xticks(x + bar_width * 2.5)
ax.set_xticklabels([task[i] for i in range(0, 5)])

# 添加图例
ax.legend( loc='center left', bbox_to_anchor=(0,0.8))
plt.savefig('five.png')
# 显示图表
plt.show()
