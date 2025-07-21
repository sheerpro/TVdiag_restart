from collections import defaultdict
import pandas as pd
import time
from tqdm import tqdm
import numpy as np

from extractor.metric_event_extractor import extract_metric_events
from extractor.trace_event_extractor import extract_trace_events
from extractor.log_event_extractor import extract_log_events
from utils import io_util



# def convert_numpy_types(obj):
#     """转换numpy类型为Python原生类型"""
#     if isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     else:
#         return obj


def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    else:
        return obj


data: dict = io_util.load('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/post-data-10.pkl')
# 将第一列设置为索引
label_df = pd.read_csv('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/allgroundtruth.csv', index_col=0)

metric_detectors = io_util.load('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/detector/metric-detector-strict-host.pkl')
trace_detectors = io_util.load('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/detector/trace-detector.pkl')


metric_events_dic = defaultdict(list)
trace_events_dic = defaultdict(list)
log_events_dic = defaultdict(list)
metric_costs, trace_costs, log_costs = [], [], []

for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    print(idx)
    chunk = data[idx]
    # extract metric events
    st = time.time()
    metric_events = []
    for pod_host, kpi_dic in chunk['metric'].items():
        kpi_events = extract_metric_events(pod_host, kpi_dic, metric_detectors[pod_host])
        metric_events.extend(kpi_events)
    metric_costs.append(time.time()-st)
    metric_events_dic[idx]=metric_events
    # extract trace events
    st = time.time()
    trace_events = extract_trace_events(chunk['trace'], trace_detectors)
    trace_events_dic[idx] = trace_events
    trace_costs.append(time.time()-st)
    # extract log events
    st = time.time()
    miner = io_util.load('./drain/aiops-drain.pkl')
    log_df = chunk['log'].astype(str)
    log_events = extract_log_events(log_df, miner, 0.5)
    log_events_dic[idx] = log_events
    log_costs.append(time.time()-st)

metric_time = np.mean(metric_costs)
trace_time = np.mean(trace_costs)
log_time = np.mean(log_costs)
print(f'the time cost of extract metric events is {metric_time}')
print(f'the time cost of extract trace events is {trace_time}')
print(f'the time cost of extract log events is {log_time}')
#the time cost of extract metric events is 0.18307018280029297
# the time cost of extract trace events is 0.23339865726162023
# the time cost of extract log events is 0.6638196256618483


# 在保存之前转换数据
metric_events_dic = {int(k): [convert_numpy_types(x) for x in v] 
                    for k, v in metric_events_dic.items()}
trace_events_dic = {int(k): [convert_numpy_types(x) for x in v] 
                    for k, v in trace_events_dic.items()}
log_events_dic = {int(k): [convert_numpy_types(x) for x in v] 
                  for k, v in log_events_dic.items()}


io_util.save_json('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/events/log.json', log_events_dic)
io_util.save_json('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/events/metric.json', metric_events_dic)
io_util.save_json('/home/fuxian/lky/TVDiag_new_aiops/extractor/AIOPS/events/trace.json', trace_events_dic)



