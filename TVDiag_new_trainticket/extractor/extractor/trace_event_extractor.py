import json
import os
import numpy as np

import pandas as pd
import utils.io_util as io_util
import utils.detect_util as d_util

# http
def slide_window(df, win_size):
    sts, ds, err_500_ps, err_400_ps=[], [], [], []
    df['duration'] = df['ed_time']-df['st_time']
    i, time_max=df['st_time'].min(), df['st_time'].max()
    while i < time_max:
        temp_df = df[(df['st_time']>=i)&(df['st_time']<=i+win_size)]
        if temp_df.empty:
            i+=win_size
            continue
        sts.append(i)
        # error_code_n = len(temp_df[~temp_df['status_code'].isin([200, 300])])
        # err_500_ps.append(len(temp_df[temp_df['status_code']==500]))
        # err_400_ps.append(len(temp_df[temp_df['status_code']==400]))
        ds.append(temp_df['duration'].mean())
        i+=win_size
    return np.array(sts), np.array(ds)


#grpc
# import numpy as np

# def slide_window(df, win_size):
#     # 初始化数据结构
#     sts = []        # 窗口起始时间
#     ds = []         # 窗口内平均耗时
#     cnt_1 = []      # CANCELLED (1) 计数
#     cnt_2 = []      # UNKNOWN (2) 计数
#     cnt_4 = []      # INVALID_ARGUMENT (4) 计数
#     cnt_9 = []      # DEADLINE_EXCEEDED (9) 计数
#     cnt_13 = []     # INTERNAL (13) 计数
#     cnt_14 = []     # UNAVAILABLE (14) 计数

#     df['duration'] = df['ed_time'] - df['st_time']
#     i, time_max = df['st_time'].min(), df['st_time'].max()
#     while i < time_max:
#         temp_df = df[(df['st_time'] >= i) & (df['st_time'] <= i + win_size)]
#         if temp_df.empty:
#             i += win_size
#             continue

#         sts.append(i)
#         # 直接统计各状态码的出现次数
#         cnt_1.append(len(temp_df[temp_df['status_code'] == 1]))
#         cnt_2.append(len(temp_df[temp_df['status_code'] == 2]))
#         cnt_4.append(len(temp_df[temp_df['status_code'] == 4]))
#         cnt_9.append(len(temp_df[temp_df['status_code'] == 9]))
#         cnt_13.append(len(temp_df[temp_df['status_code'] == 13]))
#         cnt_14.append(len(temp_df[temp_df['status_code'] == 14]))
#         ds.append(temp_df['duration'].mean())
#         i += win_size
#     # 返回所有数组（时间戳、平均耗时、各状态码计数）
#     return np.array(sts), np.array(ds),np.array(cnt_1),np.array(cnt_2),np.array(cnt_4),np.array(cnt_9),np.array(cnt_13),np.array(cnt_14)   

def extract_trace_events(df: pd.DataFrame, trace_detector: dict):
    """extract events using iforest from 
        trace dataframe
    """
    events = []
    df.sort_values(by=['timestamp'], inplace=True, ascending=True)
    # print("原始数据中的状态码分布:", df['status_code'].value_counts())
    if 'url' in df.columns:
        df['operation'] = df['url'].str.split('?').str[0]
        # df['operation'] = df['message'].str.split('/').str[-1]  # 提取gRPC方法名
        gp = df.groupby(['parent_name', 'service_name', 'operation'])
        # events = []

        win_size = 30 * 1000
        # detect events for every call
        for (src, dst, op), call_df in gp:
            name = src + '-' + dst +'-' + op
            test_df = call_df
            test_win_sts, test_durations, err_1_ps, err_2_ps,err_4_ps, err_9_ps,err_13_ps, err_14_ps = slide_window(test_df, win_size)
            # print(f"222222222222222222222222Status code counts - 开始时间: {test_win_sts}, 持续时间: {test_durations}")
            # print(f"########################Status code counts - 1: {err_1_ps}, 2: {err_2_ps},4: {err_4_ps}, 9: {err_9_ps},13: {err_13_ps}, 14: {err_14_ps}")
            # 检查slide_window返回的数据
            
            if len(test_durations) > 0:
                pd_idx = iforest(trace_detector[name]['dur_detector'], test_durations)
                err_1_idx = iforest(trace_detector[name]['1_detector'], err_1_ps)
                err_2_idx = iforest(trace_detector[name]['2_detector'], err_2_ps)
                err_4_idx = iforest(trace_detector[name]['4_detector'], err_4_ps)
                err_9_idx = iforest(trace_detector[name]['9_detector'], err_9_ps)
                err_13_idx = iforest(trace_detector[name]['13_detector'], err_13_ps)
                err_14_idx = iforest(trace_detector[name]['14_detector'], err_14_ps)


                if pd_idx != -1:
                    events.append([test_win_sts[pd_idx], src, dst, op, 'PD'])
                if err_1_idx != -1:
                    events.append([test_win_sts[err_1_idx], src, dst, op, '1'])
                if err_2_idx != -1:
                    events.append([test_win_sts[err_2_idx], src, dst, op, '2'])
                if err_4_idx != -1:
                    events.append([test_win_sts[err_4_idx], src, dst, op, '4'])
                if err_9_idx != -1:
                    events.append([test_win_sts[err_9_idx], src, dst, op, '9'])
                if err_13_idx != -1:
                    events.append([test_win_sts[err_13_idx], src, dst, op, '13'])
                if err_14_idx != -1:
                    events.append([test_win_sts[err_14_idx], src, dst, op, '14'])
                
        events = sorted(events, key=lambda x: x[0])
        events = [x[1:] for x in events]
    else:
        gp = df.groupby(['parent_name', 'service_name'])
        # events = []

        win_size = 30 * 1000
        # detect events for every call
        for (src, dst), call_df in gp:
            name = src + '-' + dst
            test_df = call_df
            test_win_sts, test_durations = slide_window(test_df, win_size)
            # print(f"222222222222222222222222Status code counts - 开始时间: {test_win_sts}, 持续时间: {test_durations}")
            # print(f"########################Status code counts - 1: {err_1_ps}, 2: {err_2_ps},4: {err_4_ps}, 9: {err_9_ps},13: {err_13_ps}, 14: {err_14_ps}")
            # print(f"########################Status code counts - 1: {err_1_ps}, 2: {err_2_ps}, 4:{err_4_ps},9:{err_9_ps},13:{err_13_ps},14:{err_14_ps}")
            if len(test_durations) > 0:
                pd_idx = iforest(trace_detector[name]['dur_detector'], test_durations)
                # err_1_idx = iforest(trace_detector[name]['1_detector'], err_1_ps)
                # err_2_idx = iforest(trace_detector[name]['2_detector'], err_2_ps)
                # err_4_idx = iforest(trace_detector[name]['4_detector'], err_4_ps)
                # err_9_idx = iforest(trace_detector[name]['9_detector'], err_9_ps)
                # err_13_idx = iforest(trace_detector[name]['13_detector'], err_13_ps)
                # err_14_idx = iforest(trace_detector[name]['14_detector'], err_14_ps)
                # 在extract_trace_events函数中添加调试输出
                # print(f"Detected anomalies for {name}:")
                # print(f"……………………PD idx: {pd_idx}, 1 idx: {err_1_idx}, 2 idx: {err_2_idx},4 idx: {err_4_idx},9 idx: {err_9_idx},13 idx: {err_13_idx},14 idx: {err_14_idx}")
                
                if pd_idx != -1:
                    events.append([test_win_sts[pd_idx], src, dst, 'PD'])
                # if err_1_idx != -1:
                #     events.append([test_win_sts[err_1_idx], src, dst, '1'])
                # if err_2_idx != -1:
                #     events.append([test_win_sts[err_2_idx], src, dst, '2'])
                # if err_4_idx != -1:
                #     events.append([test_win_sts[err_4_idx], src, dst, '4'])
                # if err_9_idx != -1:
                #     events.append([test_win_sts[err_9_idx], src, dst, '9'])
                # if err_13_idx != -1:
                #     events.append([test_win_sts[err_13_idx], src, dst, '13'])
                # if err_14_idx != -1:
                #     events.append([test_win_sts[err_14_idx], src, dst, '14'])
                
        events = sorted(events, key=lambda x: x[0])
        events = [x[1:] for x in events]
    return events


def iforest(detector, test_arr):
    """隔离森林检测（保持不变）"""
    labels = detector.predict(test_arr.reshape(-1,1)).tolist()
    try:
        idx = labels.index(-1)
    except:
        return -1
    return idx


    # labels = detector.predict(test_arr.reshape(-1,1)).tolist()
    # return [i for i, x in enumerate(labels) if x == -1]  # 返回所有异常点索引