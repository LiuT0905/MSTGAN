import pandas as pd
import numpy as np

# 读取 Excel 文件（无表头）
file_path = r"D:\QQdown\2402成绩.xlsx" # 请确保文件路径正确
df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

# 指定列索引（0-based）
col_student = 0  # 第1列
col_score = 8  # 第9列
col_credit = 9  # 第10列
col_category = 10  # 第11列

# 提取所需列并命名
data = df[[col_student, col_score, col_credit, col_category]].copy()
data.columns = ['学号', '成绩', '学分', '课程类别']

# 转换成绩和学分为数值，非数值转为 NaN
data['成绩'] = pd.to_numeric(data['成绩'], errors='coerce')
data['学分'] = pd.to_numeric(data['学分'], errors='coerce')

# 过滤有效行：成绩和学分均有效且学分 > 0
valid = data.dropna(subset=['成绩', '学分'])
valid = valid[valid['学分'] > 0]

# 定义必修和选修类别
required_categories = ['基础理论课', '专业必修课', '公共必修课', '必修环节', '必修课']
elective_categories = ['公共选修课', '专业选修课']

# 标记课程类型
valid['课程类型'] = valid['课程类别'].apply(
    lambda x: '必修' if x in required_categories else ('选修' if x in elective_categories else '其他')
)

# 只保留必修和选修
filtered = valid[valid['课程类型'].isin(['必修', '选修'])]

# 按学号分组计算
results = []
for student_id, group in filtered.groupby('学号'):
    # 必修组
    req = group[group['课程类型'] == '必修']
    # 选修组
    ele = group[group['课程类型'] == '选修']

    # 计算必修加权平均
    if len(req) > 0 and req['学分'].sum() > 0:
        req_avg = (req['成绩'] * req['学分']).sum() / req['学分'].sum()
    else:
        req_avg = 0

    # 计算选修加权平均
    if len(ele) > 0 and ele['学分'].sum() > 0:
        ele_avg = (ele['成绩'] * ele['学分']).sum() / ele['学分'].sum()
    else:
        ele_avg = 0

    s2 = req_avg * 0.7 + ele_avg * 0.3
    results.append({'学号': student_id, 'S2': round(s2, 2)})

# 输出结果
result_df = pd.DataFrame(results)
print(result_df)

# 保存结果
result_df.to_excel('学生S2计算结果.xlsx', index=False)