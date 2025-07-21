import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from utils import io_util

labels = pd.read_csv('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/allgroundtruth.csv')
failure_pre_data: dict = io_util.load('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/pre-data.pkl')


normal_metrics = {}
normal_traces = defaultdict(list)

for idx, row in tqdm(labels.iterrows(), total=labels.shape[0]):
    if row['data_type'] == 'test':
        continue
    index = row['index']
    chunk = failure_pre_data[index]
    for pod, kpi_dic in chunk['metric'].items():
        if pod not in normal_metrics.keys():
            normal_metrics[pod] = defaultdict(list)
        for kpi, kpi_df in kpi_dic.items():
            normal_metrics[pod][kpi].append(kpi_df)
            
    trace_df = chunk['trace']

    # trace_df['operation'] = trace_df['url'].str.split('?').str[0]
    if 'url' in trace_df.columns:
       
        trace_df['operation'] = trace_df['url'].str.split('?').str[0]
        trace_gp = trace_df.groupby(['parent_name', 'service_name', 'operation'])
        for (src, dst, op), call_df in trace_gp:
            name = src + '-' + dst + '-' + op
            normal_traces[name].append(call_df)
    else:
        
        trace_gp = trace_df.groupby(['parent_name', 'service_name'])
        for (src, dst), call_df in trace_gp:
            name = src + '-' + dst
            normal_traces[name].append(call_df)
            # print(normal_traces)
    # trace_df['operation'] = trace_df['message'].str.split('/').str[-1]
    # 在进行 groupby 操作前添加这行代码来查看所有列名
    # print(trace_df.columns)

for pod in normal_metrics.keys():
    for kpi, kpi_dfs in normal_metrics[pod].items():
        normal_metrics[pod][kpi] = pd.concat(kpi_dfs)

io_util.save('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/detector/normal_traces.pkl', normal_traces)
io_util.save('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/detector/normal_metrics.pkl', normal_metrics)

############################################################################

import numpy as np
from sklearn.ensemble import IsolationForest
from extractor.trace_event_extractor import slide_window
from utils import io_util
import time



normal_traces = io_util.load('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/detector/normal_traces.pkl')
normal_metrics = io_util.load('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/detector/normal_metrics.pkl')

metric_detectors = {}
for pod in normal_metrics.keys():
    metric_detectors[pod] = {}
    for kpi, dfs in normal_metrics[pod].items():
        metric_detectors[pod][kpi] = [
            normal_metrics[pod][kpi]['value'].mean(), 
            normal_metrics[pod][kpi]['value'].std()
        ]
st = time.time()
trace_detectors = {}
for name, call_dfs in normal_traces.items():
    # trace_detectors[name] = {
    #     'dur_detector': IsolationForest(random_state=0, n_estimators=5),
    #     '500_detector': IsolationForest(random_state=0, n_estimators=5),
    #     '400_detector': IsolationForest(random_state=0, n_estimators=5)
    # }
    trace_detectors[name] = {
        'dur_detector': IsolationForest(random_state=0, n_estimators=5),
        '1_detector': IsolationForest(random_state=0, n_estimators=5),
        '2_detector': IsolationForest(random_state=0, n_estimators=5),
        '4_detector': IsolationForest(random_state=0, n_estimators=5),
        '9_detector': IsolationForest(random_state=0, n_estimators=5),
        '13_detector': IsolationForest(random_state=0, n_estimators=5),
        '14_detector': IsolationForest(random_state=0, n_estimators=5)
    }
    train_ds, train_1_ep, train_2_ep,train_4_ep, train_9_ep,train_13_ep, train_14_ep = [], [], [],[], [], [],[]
    for call_df in call_dfs:
        # _, durs, err_500_ps, err_400_ps = slide_window(call_df, 30 * 1000)
        win_size = 300 * 1000
        # print("#######################")
        _, durs, err_1_ps, err_2_ps, err_4_ps, err_9_ps, err_13_ps,err_14_ps = slide_window(call_df, win_size)
        # 这个 slide_window 函数的作用是​​对调用跟踪数据（df）按时间窗口（win_size）进行滑动窗口统计​​，计算每个窗口内的平均响应时间、500错误次数和400错误次数
        train_ds.extend(durs)
        # train_500_ep.extend(err_500_ps)
        # train_400_ep.extend(err_400_ps)
        # print(f"Code 1 count: {sum(err_1_ps)}")
        # print(f"Code 2 count: {sum(err_2_ps)}")
        # print(f"Code 4 count: {sum(err_4_ps)}")
        # print(f"Code 9 count: {sum(err_9_ps)}")
        # print(f"Code 13 count: {sum(err_13_ps)}")
        # print(f"Code 14 count: {sum(err_14_ps)}")

        train_1_ep.extend(err_1_ps)
        train_2_ep.extend(err_2_ps)
        train_4_ep.extend(err_4_ps)
        train_9_ep.extend(err_9_ps)
        train_13_ep.extend(err_13_ps)
        train_14_ep.extend(err_14_ps)
        # 在构建检测器的循环中添加调试输出
        # print(f"\nStatus code distribution for {name}:")
        # print(f"Code 1 count: {sum(train_1_ep)}")
        # print(f"Code 2 count: {sum(train_2_ep)}")
        # print(f"Code 4 count: {sum(train_4_ep)}")
        # print(f"Code 9 count: {sum(train_9_ep)}")
        # print(f"Code 13 count: {sum(train_13_ep)}")
        # print(f"Code 14 count: {sum(train_14_ep)}")
    if len(train_ds) == 0:
        continue
    # dur_clf, err_500_clf, err_400_clf = trace_detectors[name]['dur_detector'], trace_detectors[name]['500_detector'], trace_detectors[name]['400_detector']
    # dur_clf.fit(np.array(train_ds).reshape(-1,1))
    # err_500_clf.fit(np.array(err_500_ps).reshape(-1,1))
    # err_400_clf.fit(np.array(err_400_ps).reshape(-1,1))

    dur_clf, err_1_clf, err_2_clf,err_4_clf, err_9_clf,err_13_clf, err_14_clf = trace_detectors[name]['dur_detector'], trace_detectors[name]['1_detector'], trace_detectors[name]['2_detector'],trace_detectors[name]['4_detector'], trace_detectors[name]['9_detector'],trace_detectors[name]['13_detector'], trace_detectors[name]['14_detector']
    dur_clf.fit(np.array(train_ds).reshape(-1,1))
    err_1_clf.fit(np.array(err_1_ps).reshape(-1,1))
    err_2_clf.fit(np.array(err_2_ps).reshape(-1,1))
    err_4_clf.fit(np.array(err_4_ps).reshape(-1,1))
    err_9_clf.fit(np.array(err_9_ps).reshape(-1,1))
    err_13_clf.fit(np.array(err_13_ps).reshape(-1,1))
    err_14_clf.fit(np.array(err_14_ps).reshape(-1,1))
    
    # trace_detectors[name]['dur_detector']=dur_clf
    # trace_detectors[name]['500_detector']=err_500_clf
    # trace_detectors[name]['400_detector']=err_400_clf
    trace_detectors[name]['dur_detector']=dur_clf
    trace_detectors[name]['1_detector']=err_1_clf
    trace_detectors[name]['2_detector']=err_2_clf
    trace_detectors[name]['4_detector']=err_4_clf
    trace_detectors[name]['9_detector']=err_9_clf
    trace_detectors[name]['13_detector']=err_13_clf
    trace_detectors[name]['14_detector']=err_14_clf

ed = time.time()
io_util.save('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/detector/trace-detector.pkl', trace_detectors)
io_util.save('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/detector/metric-detector-strict-host.pkl', metric_detectors)

print(ed-st)

