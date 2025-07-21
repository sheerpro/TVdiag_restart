from collections import defaultdict
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

from extractor.metric_event_extractor import extract_metric_events
from extractor.trace_event_extractor import extract_trace_events
from extractor.log_event_extractor import extract_log_events
from utils import io_util


failure_post_data: dict = io_util.load('/home/fuxian/lky/TVDiag_new/extractor/AIOPS/post-data-10.pkl')
# 将第一列设置为索引
label_df = pd.read_csv('/home/fuxian/lky/TVDiag_new/extractor/AIOPS/allgroundtruth.csv', index_col=0)

for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    chunk = failure_post_data[idx]
    # 对每个label提取依赖关系（服务之间是调用关系，同一个节点的服务为双向边）
    trace_df = chunk['trace']
    svcs = []
    influences = []

    # 捕获服务与节点的对应关系
    node2svcs = defaultdict(list)
    all_cmdb_ids = set()

    # 遍历所有日期目录
    base_dir = '/home/fuxian/lky/TVDiag_new/extractor/AIOPS'
    for date_dir in sorted(os.listdir(base_dir)):
        # 检查是否是日期格式的目录
        if not os.path.isdir(os.path.join(base_dir, date_dir)) or not date_dir.startswith('2022-'):
            continue
            
        metric_dir = os.path.join(base_dir, date_dir, "metric")
        if not os.path.exists(metric_dir):
            print(f"警告: {date_dir} 下缺少metric目录")
            continue
            
        # 遍历该日期目录下的所有metric文件
        for metric_file in os.listdir(metric_dir):
            if not metric_file.endswith('.csv'):
                continue
                
            try:
                metric_df = pd.read_csv(os.path.join(metric_dir, metric_file))
                all_cmdb_ids.update(metric_df['cmdb_id'].unique())
            except Exception as e:
                print(f"处理文件 {metric_file} 时出错: {str(e)}")
                continue

        
    for node, pods in node2svcs.items():
        # print(node)
        # print(pods)
        # print('============================')
        svcs.extend(pods)
        for i in range(len(pods)):
            for j in range(i + 1, len(pods)):
                influences.append([pods[i], pods[j]]) 
                influences.append([pods[j], pods[i]])
    svcs = list(set(svcs))
    svcs.sort()

def process_service_relationships(trace_df, node2svcs):
    # 添加调试信息
    print("从node2svcs中获取的服务:")
    for node, services in node2svcs.items():
        print(f"Node {node}: {services}")
    
    # 收集所有服务名称
    svcs = set()
    influences = []
    
    # 1. 首先添加node2svcs中的所有服务
    metric_services = set()
    for pods in node2svcs.values():
        metric_services.update(pods)
    print(f"Metric中的服务: {metric_services}")  # 调试信息
    svcs.update(metric_services)
    
    # 2. 添加trace数据中的服务
    trace_services = set(trace_df['service_name'].dropna().unique()) | set(trace_df['parent_name'].dropna().unique())
    print(f"Trace中的服务: {trace_services}")  # 调试信息
    svcs.update(trace_services)
    
    # 转换为排序列表并保留所有服务
    svcs = sorted(list(svcs))
    print(f"最终服务列表: {svcs}")  # 调试信息
    
    # 处理同一节点上服务之间的影响关系
    for node, pods in node2svcs.items():
        for i in range(len(pods)):
            for j in range(i + 1, len(pods)):
                influences.append([pods[i], pods[j]])
                influences.append([pods[j], pods[i]])
    
    # 捕获服务调用关系
    edge_columns = ['service_name', 'parent_name']
    calls = trace_df.dropna(subset=['parent_name']).drop_duplicates(subset=edge_columns)[edge_columns].values.tolist()
    
    # 合并所有关系并去重
    all_relationships = pd.DataFrame(calls + influences).drop_duplicates().values.tolist()
    
    # 构建边列表
    edges = []
    for source, target in all_relationships:
        try:
            source_idx = svcs.index(source)
            target_idx = svcs.index(target)
            edges.append([source_idx, target_idx])
        except ValueError as e:
            print(f"警告: 找不到服务 {source} 或 {target}")
            continue
            
    return svcs, edges



############################################################################################################################
from tqdm import tqdm

# 创建存储最终结果的字典
deployment_nodes = {}
deployment_edges = {}

# 修改主循环
for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    chunk = failure_post_data[idx]
    trace_df = chunk['trace']
    
    # 处理服务关系
    svcs, edge_list = process_service_relationships(trace_df, node2svcs)
    
    # 存储结果
    deployment_nodes[str(idx)] = svcs
    deployment_edges[str(idx)] = edge_list
    
    # 更新chunk（如果需要）
    chunk['nodes'] = svcs
    chunk['edges'] = edge_list



io_util.save_json('/home/fuxian/lky/TVDiag_new/extractor/AIOPS/events/nodes.json', deployment_nodes)
io_util.save_json('/home/fuxian/lky/TVDiag_new/extractor/AIOPS/events/edges.json', deployment_edges)