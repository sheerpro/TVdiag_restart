import csv
from collections import Counter

# 文件路径
csv_file_path = '/home/fuxian/lky/TVDiag_new/extractor/GAIA/2021-07-04/trace/trace.csv'  # 替换为你的 CSV 文件路径

# 初始化计数器
status_code_counter = Counter()

# 读取 CSV 文件并统计 status_code 的值
try:
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if 'status_code' in row:  # 确保键存在
                status_code_counter[row['status_code']] += 1
            else:
                print("CSV 文件中未找到 'status_code' 键")
                break
except FileNotFoundError:
    print(f"文件 {csv_file_path} 未找到，请检查路径")
except Exception as e:
    print(f"发生错误: {e}")

# 输出统计结果
for status_code, count in status_code_counter.items():
    print(f"状态码 {status_code}: 出现 {count} 次")