import numpy as np
import numpy as np
import pandas as pd

# 设置随机数种子，以便每次运行代码时都生成相同的随机数
np.random.seed(0)

# 生成15行100列的数据，数据值在200到1000之间
data = np.random.randint(200, 1000, size=(20, 100))

# 创建DataFrame
df = pd.DataFrame(data)

# 在DataFrame前面插入一行空值，首先创建一个全为空值的行
empty_row = pd.DataFrame(np.nan, index=[0], columns=df.columns)

# 将空行与原数据合并，使第一行为空
df = pd.concat([empty_row, df], ignore_index=True)

# 保存到Excel表格
excel_file = '2000task200-1000.xlsx'
df.to_excel(excel_file, index=False, header=False)

print(f"数据已保存到Excel文件：{excel_file}")