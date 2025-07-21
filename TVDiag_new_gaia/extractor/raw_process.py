import gc
import psutil
import os

import numpy as np
from tqdm import tqdm
from utils import io_util
from utils.time_util import *
import pandas as pd
import time
import random
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

random.seed(12)
np.random.seed(12)

def process_traces(dir):
    """处理单个目录下的trace数据
    Args:
        dir: trace文件所在目录
    Returns:
        处理后的DataFrame，包含parent_name列
    """
    print(f"读取trace数据从: {dir}")
    trace_file = os.path.join(dir, "trace.csv")
    
    if not os.path.exists(trace_file):
        raise FileNotFoundError(f"找不到trace文件: {trace_file}")
        
    trace_df = pd.read_csv(trace_file)
    
    # 打印所有列名
    # print("\ntrace_df的列名:")
    # print(trace_df.columns.tolist())
    
    # 确保必要的列存在
    required_columns = ['span_id', 'service_name', 'parent_id', 'st_time']
    missing_columns = [col for col in required_columns if col not in trace_df.columns]
    if missing_columns:
        raise ValueError(f"Trace数据缺少必要的列: {missing_columns}")
    
    
    # 添加处理parent_name
    trace_df = spans_df_left_join(trace_df)
    
    # 验证处理结果
    if 'parent_name' not in trace_df.columns:
        print("处理后的列名:", trace_df.columns.tolist())
        raise ValueError("spans_df_left_join处理后缺少parent_name列")
    return trace_df


def spans_df_left_join(spans_df_ori_1: pd.DataFrame) -> pd.DataFrame:
    """加入parent_name属性"""
    try:
        # 检查必要的列
        required_columns = ['span_id', 'service_name', 'parent_id']
        for col in required_columns:
            if col not in spans_df_ori_1.columns:
                raise ValueError(f"输入数据缺少必要的列: {col}")
        
        spans_df_temp = spans_df_ori_1.copy()
        # 只保留需要的列
        spans_df_ori_1 = spans_df_ori_1[['span_id', 'service_name']].copy()
        # 重命名
        spans_df_ori_1.rename(columns={'service_name': 'parent_name'}, inplace=True)
        
        # 打印合并前的信息
        print(f"原始数据行数: {len(spans_df_temp)}")
        print(f"用于合并的数据行数: {len(spans_df_ori_1)}")
        
        # 执行合并并验证结果
        spans_df_temp = spans_df_temp.merge(
            spans_df_ori_1, 
            left_on='parent_id', 
            right_on='span_id', 
            how='left'
        )
        
        # 打印合并后的信息和验证
        print(f"合并后的行数: {len(spans_df_temp)}")
        print(f"parent_name为空的行数: {spans_df_temp['parent_name'].isna().sum()}")
        print("合并后的列名:", spans_df_temp.columns.tolist())
        
        # 清理列名
        if 'span_id_x' in spans_df_temp.columns:
            spans_df_temp.rename(columns={'span_id_x': 'span_id'}, inplace=True)
        if 'span_id_y' in spans_df_temp.columns:
            spans_df_temp.drop(columns=['span_id_y'], inplace=True)
            
        # 验证parent_name列是否存在
        if 'parent_name' not in spans_df_temp.columns:
            raise ValueError("合并后的DataFrame中缺少parent_name列")
            
        return spans_df_temp
        
    except Exception as e:
        print(f"在spans_df_left_join中发生错误: {str(e)}")
        raise

def process_logs(dir):
    def extract_Date(df: pd.DataFrame):
        df.dropna(axis=0, subset=['message'], inplace=True)
        df['timestamp'] = df['message'].map(lambda m: m.split(',')[0])
        df['timestamp'] = df['timestamp'].apply(lambda x: time2stamp(str(x)))
        return df

    dfs = []
    for f in os.listdir(dir):
        if f.endswith("2021-07.csv"):
            df = pd.read_csv(f"{dir}/{f}")
            df = extract_Date(df)
            dfs.append(df)
    log_df = pd.concat(dfs)
    log_df.to_csv("log.csv")


def extract_traces(trace_df: pd.DataFrame, start_time):
    window = 10 * 60 * 1000
    trace_df['st_time'] = trace_df['st_time'].apply(convert_to_milliseconds)
    con1 = trace_df['st_time'] > start_time - 4*window
    con2 = trace_df['st_time'] < start_time
    con3 = trace_df['st_time'] > start_time
    con4 = trace_df['st_time'] < start_time + window
    return trace_df[con1 & con2], trace_df[con3 & con4]


def extract_logs(log_df: pd.DataFrame, start_time):
    window = 10 * 60 * 1000
    log_df['timestamp'] = log_df['timestamp'].apply(convert_to_milliseconds)
    con1 = log_df['timestamp'] > start_time - 4*window
    con2 = log_df['timestamp'] < start_time
    con3 = log_df['timestamp'] > start_time
    con4 = log_df['timestamp'] < start_time + window
    return log_df[con1 & con2], log_df[con3 & con4]

def extract_metrics(metric_df: pd.DataFrame, start_time):
    window = 10 * 60 * 1000
    # 确保timestamp列是数值类型
    metric_df['timestamp'] = pd.to_numeric(metric_df['timestamp'])
    metric_df['timestamp'] = metric_df['timestamp'].apply(convert_to_milliseconds)
    con1 = metric_df['timestamp'] > start_time - 4*window
    con2 = metric_df['timestamp'] < start_time
    con3 = metric_df['timestamp'] > start_time
    con4 = metric_df['timestamp'] < start_time + window
    return metric_df[con1 & con2], metric_df[con3 & con4]

def process_all_logs(base_dir):
    """只处理所有日期目录下的log数据"""
    print(f"开始处理所有日期的log数据...")
    all_logs = []
    
    for date_dir in sorted(os.listdir(base_dir)):
        # 检查是否是日期格式的目录
        if not os.path.isdir(os.path.join(base_dir, date_dir)) or not date_dir.startswith('2021-'):
            continue
            
        log_dir = os.path.join(base_dir, date_dir, "log")
        if not os.path.exists(log_dir):
            print(f"警告: {date_dir} 下缺少log目录")
            continue
            
        log_file = os.path.join(log_dir, "log.csv")
        if not os.path.exists(log_file):
            print(f"警告: {date_dir}/log 下缺少log.csv文件")
            continue
            
        print(f"\n处理 {date_dir} 的log数据...")
        try:
            df = pd.read_csv(log_file)
            all_logs.append(df)
            print(f"成功处理 {date_dir} 的log数据，行数: {len(df)}")
        except Exception as e:
            print(f"处理 {date_dir} 时出错: {str(e)}")
            continue
    
    if not all_logs:
        raise ValueError("没有成功处理任何log数据")
        
    final_log_df = pd.concat(all_logs, ignore_index=True)
    print(f"\n完成所有log数据处理，总行数: {len(final_log_df)}")
    return final_log_df

def process_all_traces(base_dir):
    """只处理所有日期目录下的trace数据"""
    print(f"开始处理所有日期的trace数据...")
    all_traces = []
    
    for date_dir in sorted(os.listdir(base_dir)):
        # 检查是否是日期格式的目录（如2021-07-01）
        if not os.path.isdir(os.path.join(base_dir, date_dir)) or not date_dir.startswith('2021-'):
            continue
            
        trace_dir = os.path.join(base_dir, date_dir, "trace")
        if not os.path.exists(trace_dir):
            print(f"警告: {date_dir} 下缺少trace目录")
            continue
            
        print(f"\n处理 {date_dir} 的trace数据...")
        try:
            trace_df = process_traces(trace_dir)
            trace_df['date'] = date_dir  # 添加日期标记
            all_traces.append(trace_df)
            print(f"成功处理 {date_dir} 的trace数据，行数: {len(trace_df)}")
        except Exception as e:
            print(f"处理 {date_dir} 时出错: {str(e)}")
            continue
    
    if not all_traces:
        raise ValueError("没有成功处理任何trace数据")
        
    final_trace_df = pd.concat(all_traces, ignore_index=True)
    return final_trace_df

def read_all_metrics(base_dir):
    """读取所有日期目录下的metric数据
    Args:
        base_dir: 根目录路径，包含多个日期子目录
    Returns:
        data: {
            pod_name: {
                metric_name: DataFrame
            }
        }
    """
    data = {}
    metric_dfs = {}
    
    # 遍历所有日期目录
    for date_dir in sorted(os.listdir(base_dir)):
        date_path = os.path.join(base_dir, date_dir)
        if not os.path.isdir(date_path):
            continue
            
        metric_dir = os.path.join(date_path, "metric")
        if not os.path.exists(metric_dir):
            print(f"警告: {date_dir} 下缺少metric目录")
            continue
            
        print(f"处理 {date_dir} 的metric数据...")
        
        # 读取该日期目录下的所有metric文件
        for metric_file in os.listdir(metric_dir):
            if not metric_file.endswith('.csv'):
                continue
                
            try:
                file_path = os.path.join(metric_dir, metric_file)
                df = pd.read_csv(file_path)
                df['date'] = date_dir  # 添加日期标记
                
                metric_name = metric_file.rsplit('.', 1)[0]  # 去掉.csv后缀
                
                # 将数据添加到临时字典
                if metric_name not in metric_dfs:
                    metric_dfs[metric_name] = []
                metric_dfs[metric_name].append(df)
                
            except Exception as e:
                print(f"处理文件 {metric_file} 时出错: {str(e)}")
                continue
    
    print("合并所有日期的数据...")
    
    # 合并所有日期的数据并按pod分类
    for metric_name, dfs in metric_dfs.items():
        if not dfs:  # 跳过空列表
            continue
            
        # 合并同一指标的所有日期数据
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # 根据cmdb_id分配到对应的pod
        for pod in merged_df['cmdb_id'].unique():
            pod_data = merged_df[merged_df['cmdb_id'] == pod].copy()
            
            # 初始化数据结构
            if pod not in data:
                data[pod] = {}
            
            # 存储数据
            data[pod][metric_name] = pod_data
    
    print("完成所有metric数据的处理")
    return data

def convert_to_milliseconds(timestamp):
    """将时间戳统一转换为毫秒格式"""
    if timestamp < 1e11:  # 如果是秒级时间戳
        return int(timestamp * 1000)
    return int(timestamp)

def check_time_range(trace_df, log_df, label_df):
    # 统一转换为毫秒级时间戳
    trace_min = convert_to_milliseconds(trace_df['st_time'].min())
    trace_max = convert_to_milliseconds(trace_df['st_time'].max())
    log_min = convert_to_milliseconds(log_df['timestamp'].min())
    log_max = convert_to_milliseconds(log_df['timestamp'].max())
    label_min = label_df['st_time'].min()
    label_max = label_df['st_time'].max()
    
    print("\n时间范围检查 (毫秒):")
    print(f"Trace: {trace_min} - {trace_max}")
    print(f"Log: {log_min} - {log_max}")
    print(f"Label: {label_min} - {label_max}")
    
    if label_min < trace_min or label_max > trace_max:
        print("警告: 标签时间范围超出Trace数据范围")
    if label_min < log_min or label_max > log_max:
        print("警告: 标签时间范围超出Log数据范围")


if __name__ == '__main__':
    # 处理 trace 数据并保存带有 parent_name 列的版本
    input_dir = "/home/fuxian/lky/TVDiag_new/extractor/GAIA"
    output_dir = "/home/fuxian/lky/TVDiag_new/extractor/GAIA"
    trace_df = process_all_traces(input_dir)

    # 保存处理后的数据
    processed_trace_file = os.path.join(output_dir, "all_processed_traces.csv")
    trace_df.to_csv(processed_trace_file, index=False)

    
    # 读取处理后的数据
    trace_df = pd.read_csv(processed_trace_file)
    log_df = process_all_logs(input_dir)

    label_df = pd.read_csv("/home/fuxian/lky/TVDiag_new/extractor/GAIA/allgroundtruth.csv")
    label_df['st_time'] = label_df['st_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))
    label_df['ed_time'] = label_df['ed_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))

    if 'parent_name' not in trace_df.columns:
        raise ValueError("Trace数据中缺少parent_name列,请检查process_traces函数的处理")
    
    check_time_range(trace_df, log_df, label_df)

    # 处理指标数据
    pre_data, post_data = {}, {}
    metric_data = read_all_metrics(input_dir)
    
    # 处理每个标签
    for _, row in label_df.iterrows():
        start_time = time.time()
        idx = row['index']
        pre_data[idx], post_data[idx] = {}, {}
        st_time, ed_time = row['st_time'], row['ed_time']
        pre_trace_df, post_trace_df = extract_traces(trace_df, st_time)
        pre_data[idx]['trace'] = pre_trace_df
        post_data[idx]['trace'] = post_trace_df

        pre_log_df, post_log_df = extract_logs(log_df, st_time)
        pre_data[idx]['log'] = pre_log_df
        post_data[idx]['log'] = post_log_df

        # 并行处理metric
        results = []
        # with mp.Pool(processes=4) as pool:
        #     for f in os.listdir("metric"):
        #         if f.endswith("07-15.csv"):
        #             df1 = pd.read_csv(f"metric/{f}")
        #             next_name = f.replace(
        #                 "2021-07-01_2021-07-15",
        #                 "2021-07-15_2021-07-31"
        #             )
        #             df2 = pd.read_csv(f"metric/{next_name}")
        #             metric_df = pd.concat([df1, df2])

        #             metric_name = f.split("_2021")[0]
        #             metric_df.rename(columns={"value": metric_name}, inplace=True)

        #             result = pool.apply_async(extract_metrics, [metric_df, st_time])
        #             results.append(result)
        #     [result.wait() for result in results]
        pre_metrics, post_metrics = {}, {}
        for pod, metric_dic in metric_data.items():
            pre_metrics[pod], post_metrics[pod] = {}, {}
            for metric_name, metric_df in metric_dic.items():
                pre_metrics[pod][metric_name], post_metrics[pod][metric_name] = extract_metrics(metric_df, st_time)

        pre_data[idx]['metric'] = pre_metrics
        post_data[idx]['metric'] = post_metrics

        end_time = time.time()
        process_time = end_time - start_time
        print(fr"完成{idx}, 用时{process_time}")

    # 使用os.path.join来正确拼接路径
    io_util.save(os.path.join(output_dir, "pre-data.pkl"), pre_data)
    io_util.save(os.path.join(output_dir, "post-data-10.pkl"), post_data)
