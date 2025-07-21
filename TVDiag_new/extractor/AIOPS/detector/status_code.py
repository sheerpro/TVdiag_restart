import json
from collections import Counter

def analyze_status_codes(data):
    """分析 status_code 的分布情况"""
    # 收集所有的 status_code
    status_codes = []
    for service_data in data.values():  # 遍历每个服务的数据
        for trace in service_data:  # 遍历服务中的每条记录
            if isinstance(trace, list):  # 处理嵌套列表
                for item in trace:
                    if isinstance(item, dict) and 'status_code' in item:
                        status_codes.append(item['status_code'])
            elif isinstance(trace, dict) and 'status_code' in trace:
                status_codes.append(trace['status_code'])

    # 使用 Counter 统计次数
    counter = Counter(status_codes)
    
    # 打印统计结果
    print("\nstatus_code 统计结果:")
    print("-" * 40)
    print("状态码\t\t出现次数\t占比")
    print("-" * 40)
    
    total = sum(counter.values())
    for code, count in counter.most_common():
        percentage = (count / total) * 100
        print(f"{code:<15} {count:<10} {percentage:.2f}%")

    return counter

if __name__ == "__main__":
    # 读取数据
    try:
        with open('normal_traces.json', 'r') as f:
            data = json.load(f)
        
        # 分析数据
        stats = analyze_status_codes(data)
        
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")