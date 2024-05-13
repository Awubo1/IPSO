import pandas as pd

# 定义点的数据
data = {
    'X': [0.3, 0, 1],
    'Y': [0, 0, 0]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 将DataFrame保存到Excel文件
excel_file = 'fuwuqi4.xlsx'
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet1')

print(f"数据已保存到Excel文件：{excel_file}")
