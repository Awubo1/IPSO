import pandas as pd

# 指定Excel文件路径
excel_file = 'D:\边缘计算\ipso\ipso\task.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你想要统计的工作表名称

# 使用pandas加载特定的工作表
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# 使用count()方法计算每列的非空数据数量，然后求和得到总的非空数据数量
total_non_null_data = df.count().sum()

# 输出结果
print(f"工作表 '{sheet_name}' 中非空数据的总数为：{total_non_null_data}")
