import pickle
import json
import pandas as pd
import numpy as np

def convert_to_serializable(obj):
    """将不可序列化的对象转换为可序列化的格式"""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return obj

try:
    # 读取 pkl 文件
    with open('/home/fuxian/lky/TVDiag_new_trainticket/extractor/trainticket/post-data-10.pkl', 'rb') as f:
        data = pickle.load(f)
        print("成功读取PKL文件")

    # 转换数据
    serializable_data = convert_to_serializable(data)
    print("数据转换完成")

    # 保存为 JSON
    with open('post-data-10.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=4, ensure_ascii=False)
    print("成功保存为JSON文件")

except Exception as e:
    print(f"错误: {str(e)}")