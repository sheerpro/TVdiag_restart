import json
import os
import pandas as pd


# def extract_metric_events(pod_host: str, kpi_dic: dict, metric_detector: dict):
#     """extract events using 3-sigma from 
#         different metric dataframe
#     """
#     print("111111111111",pod_host)
#     events = []
#     for kpi, df in kpi_dic.items():
#         df.fillna(0, inplace=True)
#         df.sort_values(by=['timestamp'], inplace=True, ascending=True)

#         times = df['timestamp'].values
#         if len(df)==0:
#             continue
#         # detect anomaly using 3-sigma
#         ab_idx, ab_direction = k_sigma(
#             detector=metric_detector[kpi],
#             test_arr=df['value'].values,
#             k=3,
#         )
        
#         if ab_idx != -1:
#             ab_t = times[ab_idx]
#             splits = pod_host.split('_')
#             pod, host = splits[0], splits[1]
#             events.append([ab_t, pod, host, kpi, ab_direction])
            
#     # sort by timestamp
#     sorted_events = sorted(events, key=lambda e:e[0])
#     # remove timestamp
#     sorted_events = [e[1:] for e in sorted_events]
#     return sorted_events




def extract_metric_events(pod_host: str, kpi_dic: dict, metric_detector: dict):
    """从不同指标数据框中提取使用3-sigma的事件"""
    events = []
    
    # try:
    for kpi, df in kpi_dic.items():
        
        df.fillna(0, inplace=True)
        df.sort_values(by=['timestamp'], inplace=True, ascending=True)

        times = df['timestamp'].values
        if len(df) == 0:
            continue
            
        # 使用3-sigma检测异常
        ab_idx, ab_direction = k_sigma(
            detector=metric_detector[kpi],
            test_arr=df['value'].values,
            k=3,
        )
        
        if ab_idx != -1:
            ab_t = times[ab_idx]
            # 增加错误处理
            # if '_' not in pod_host:
            #     # print(f"警告: pod_host '{pod_host}' 格式不正确，应该包含下划线分隔符")
            #     pod = pod_host
            #     host = "unknown"
            # else:
            #     pod, host = pod_host.split('_')

            if '.' in pod_host:
                parts = pod_host.split('.')
                if len(parts) > 1:
                    pod = parts[-1]
                    host = '.'.join(parts[:-1])  # 合并除最后一部分的所有部分作为 host
                else:
                    pod = pod_host
                    host = "unknown"
            pod = pod_host
            host = "unknown"
            kpi = kpi[:-4] if kpi.endswith('.csv') else kpi
            events.append([ab_t, pod, host, kpi, ab_direction])
    # sort by timestamp
    sorted_events = sorted(events, key=lambda e:e[0])
    # remove timestamp
    sorted_events = [e[1:] for e in sorted_events]
        
    # except Exception as e:
    #     print(f"处理 pod_host '{pod_host}' 时发生错误: {str(e)}")
        
    # 始终返回事件列表，即使是空的
    return sorted_events


# def extract_metric_events(pod_host: str, kpi_dic: dict, metric_detector: dict):
#     """提取并格式化metric事件为：pod, hostip, 指标名, 异常方向"""
#     events = []
    
#     try:
#         for kpi, df in kpi_dic.items():
#             df.fillna(0, inplace=True)
#             df.sort_values(by=['timestamp'], inplace=True, ascending=True)

#             if len(df) == 0:
#                 continue
                
#             # 获取当前KPI的均值和标准差
#             mean, std = metric_detector[kpi][0], metric_detector[kpi][1]
#             values = df['value'].values
            
#             # 计算3-sigma阈值
#             upper = mean + 3 * std
#             lower = mean - 3 * std
            
#             # 检测异常点
#             for idx, value in enumerate(values):
#                 if value > upper:
#                     direction = 'up'
#                     break
#                 elif value < lower:
#                     direction = 'down'
#                     break
#             else:
#                 continue  # 无异常则跳过
                
#             # 解析pod和hostip
#             if '_' in pod_host:
#                 pod, hostip = pod_host.split('_', 1)  # 只分割第一个下划线
#             else:
#                 pod, hostip = pod_host, "0.0.0.0"  # 默认IP
                
#             # 生成格式化事件
#             events.append([pod, hostip, kpi, direction])
            
#     except Exception as e:
#         print(f"Error processing {pod_host}: {str(e)}")
        
#     return events



def k_sigma(detector, test_arr, k=3):
    mean = detector[0]
    std = detector[1]
    up, lb=mean+k*std, mean-k*std

    for idx, v in enumerate(test_arr.tolist()):
        if v > up:
            return idx, 'up'
        elif v < lb:
            return idx, 'down'
    
    return -1, None
