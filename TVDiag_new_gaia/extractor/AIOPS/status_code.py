import json
from collections import defaultdict

def count_status_codes(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # 初始化计数器
    status_counts = defaultdict(int)
    
    # 遍历所有trace数据
    for trace_group in data.values():
        for span in trace_group['trace']:
            status_code = span['status_code']
            status_counts[status_code] += 1
    
    return dict(status_counts)

# 使用示例
if __name__ == "__main__":
    json_file_path = "/home/fuxian/lky/TVDiag_new_gaia/extractor/AIOPS/post-data-10.json"  # 替换为你的JSON文件路径
    status_counts = count_status_codes(json_file_path)
    
    print("Status Code统计结果:")
    for code, count in status_counts.items():
        print(f"{code}: {count}次")