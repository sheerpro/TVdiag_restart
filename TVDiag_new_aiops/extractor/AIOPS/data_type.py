import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def process_groundtruth_data():
    # 1. 读取所有groundtruth.csv文件
    base_dir = "/home/fuxian/DataSet/NewDataset/aiops"
    all_dfs = []
    
    # 遍历aiops目录下所有日期子目录
    for date_dir in sorted(os.listdir(base_dir)):  # 按日期排序
        date_path = os.path.join(base_dir, date_dir)
        if os.path.isdir(date_path):
            gt_file = os.path.join(date_path, "groundtruth.csv")
            if os.path.exists(gt_file):
                try:
                    df = pd.read_csv(gt_file)
                    df['date'] = date_dir  # 添加日期列
                    all_dfs.append(df)
                except Exception as e:
                    print(f"读取 {gt_file} 失败: {str(e)}")
    
    if not all_dfs:
        raise FileNotFoundError("未找到任何有效的groundtruth.csv文件")
    
    # 2. 合并所有数据
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 3. 添加从0开始的索引列作为第一列
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.index.name = 'index'  # 设置索引列名称
    combined_df = combined_df.reset_index()  # 将索引转为普通列
    
    # 4. 随机划分train/test (80%/20%)
    train_df, test_df = train_test_split(combined_df, test_size=0.5, random_state=42)
    train_df['data_type'] = 'train'
    test_df['data_type'] = 'test'
    combined_df = pd.concat([train_df, test_df])
    
    # 5. 调整列顺序，确保index是第一列
    cols = ['index'] + [col for col in combined_df if col != 'index']
    combined_df = combined_df[cols]
    
    # 6. 按照index列从小到大排序
    combined_df = combined_df.sort_values('index')
    
    # 7. 保存处理后的数据
    output_dir = "/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "allgroundtruth.csv")
    combined_df.to_csv(output_path, index=False)  # 不保存pandas默认索引
    
    print(f"数据处理完成，结果已保存到 {output_path}")
    print(f"总记录数: {len(combined_df)} (train: {len(train_df)}, test: {len(test_df)})")
    
    return combined_df

# 执行处理
try:
    result_df = process_groundtruth_data()
    print("\n处理后的数据样例:")
    print(result_df.head())
except Exception as e:
    print(f"处理失败: {str(e)}")