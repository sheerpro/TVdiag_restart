import json

# 读取输入 JSON 文件
with open('edges2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将第一维和第二维的数据按相同索引位置组合成二元组
result = [[data[0][i], data[1][i]] for i in range(len(data[0]))]

# 将结果保存为 JSON 文件
with open('edges3.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4)

print("转换完成，结果已保存到 output.json")