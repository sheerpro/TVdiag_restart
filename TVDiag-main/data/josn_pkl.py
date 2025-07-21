import json
import pickle

# 输入 JSON 文件路径
input_json_file = '/home/fuxian/lky/TVDiag-main/data/D3/edges3.json'  # 请替换为实际的 JSON 文件路径
# 输出 PKL 文件路径
output_pkl_file = '/home/fuxian/lky/TVDiag-main/data/D3/edges3.pkl'  # 请替换为输出的 PKL 文件路径

# 读取 JSON 文件
with open(input_json_file, 'r') as f:
    data = json.load(f)

# 保存为 PKL 文件
with open(output_pkl_file, 'wb') as f:
    pickle.dump(data, f)

print(f"已将 JSON 文件转换为 PKL 文件: {output_pkl_file}")
