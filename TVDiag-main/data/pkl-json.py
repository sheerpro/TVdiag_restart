# import pickle
# import json
# import numpy as np
# import dgl

# def pkl_to_json(pkl_file, json_file):
#     try:
#         # 读取 .pkl 文件
#         with open(pkl_file, 'rb') as f:
#             data = pickle.load(f)
        
#         # 将 ndarray 转换为列表
#         def convert_ndarray(obj):
#             if isinstance(obj, np.ndarray):
#                 return obj.tolist()  # 将 ndarray 转换为列表
#             return obj

#         # 使用递归的方式处理 data 中的所有元素
#         data = json.loads(json.dumps(data, default=convert_ndarray))

#         # 将数据写入 .json 文件
#         with open(json_file, 'w', encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False, indent=4)
        
#         print(f"成功将 {pkl_file} 转换为 {json_file}")
#     except Exception as e:
#         print(f"转换失败: {e}")

# # 示例调用
# pkl_file = '/home/fuxian/ART-master/data/D2/samples/samples.pkl'  # 输入的 pkl 文件路径
# json_file = '/home/fuxian/ART-master/data/D2/samples/samples.json'  # 输出的 json 文件路径
# pkl_to_json(pkl_file, json_file)



