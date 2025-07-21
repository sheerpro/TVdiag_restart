import pickle
import os
import numpy as np
import torch
import dgl

def load_pkl(pkl_path):
    """加载PKL文件"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"加载PKL文件时出错: {str(e)}")
        raise

def convert_to_str(obj, indent=0):
    """将对象转换为字符串表示"""
    indent_str = '  ' * indent
    if isinstance(obj, (dict)):
        result = '\n'
        for k, v in obj.items():
            result += f"{indent_str}{k}: {convert_to_str(v, indent+1)}\n"
        return result
    elif isinstance(obj, (list, tuple)):
        result = '\n'
        for item in obj:
            result += f"{indent_str}- {convert_to_str(item, indent+1)}\n"
        return result
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return str(obj.tolist())
    elif isinstance(obj, dgl.DGLGraph):
        return f"DGLGraph(nodes={obj.num_nodes()}, edges={obj.num_edges()})"
    else:
        return str(obj)

def save_txt(data, txt_path):
    """保存为TXT文件"""
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(convert_to_str(data))
        print(f"保存成功: {txt_path}")
    except Exception as e:
        print(f"保存TXT文件时出错: {str(e)}")
        raise

def pkl_to_txt(pkl_path, txt_path):
    """将PKL文件转换为TXT"""
    try:
        data = load_pkl(pkl_path)
        save_txt(data, txt_path)
    except Exception as e:
        print(f"转换失败: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    pkl_file = "/home/fuxian/lky/TVDiag_new_copy2/data/gaia/tmp/log.pkl"
    txt_file = "/home/fuxian/lky/TVDiag_new_copy2/data/gaia/tmp/log.txt"
    
    if os.path.exists(pkl_file):
        pkl_to_txt(pkl_file, txt_file)
    else:
        print(f"错误: 文件 {pkl_file} 不存在")
