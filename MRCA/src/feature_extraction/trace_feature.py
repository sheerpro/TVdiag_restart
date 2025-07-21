import pandas as pd
from typing import Dict, List

class TraceFeatureExtractor:
    def __init__(self, config: Dict):
        self.config = config
        
    def extract_latency(self, trace_path: str) -> pd.DataFrame:
        """从追踪数据中提取延迟特征"""
        # 读取数据
        traces = pd.read_csv(trace_path)
        
        # 添加调试信息
        # print("Original columns:", traces.columns.tolist())
        # print("Original data shape:", traces.shape)
        
        # 确保所需列存在
        required_columns = ['service_name', 'st_time', 'ed_time']
        if not all(col in traces.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")
        
        # 计算持续时间
        traces['duration'] = traces['ed_time'] - traces['st_time']
        
        # 预处理时间戳
        if self.config["data"].get("timestamp_format") == "unix":
            traces['st_time'] = pd.to_numeric(traces['st_time'], errors='coerce')
            traces = traces.dropna(subset=['st_time'])
        
        # 检查时间窗口配置
        if "time_window" not in self.config:
            raise ValueError("Missing 'time_window' in config")
        window = self.config["time_window"]
        
        # 创建时间窗口列
        traces['window'] = (traces['st_time'] // window * window).astype(int)
        
        # 第一次聚合：按服务和时间窗口聚合
        latencies = traces.groupby(['service_name', 'window']).agg({
            'duration': ['mean', 'max', 'min', 'std']
        }).reset_index()
        
        # 重命名列
        latencies.columns = [
            'service_name' if col[0] == 'service_name' 
            else 'window' if col[0] == 'window'
            else f'duration_{col[1]}' 
            for col in latencies.columns
        ]
        
        # 添加调试信息
        # print("\nAfter initial aggregation and renaming:")
        # print("Columns:", latencies.columns.tolist())
        # print("Data sample:\n", latencies.head())
        
        try:
            # 确保必要的列存在
            required_cols = ['window', 'service_name', 'duration_mean', 'duration_max', 'duration_min', 'duration_std']
            missing_cols = [col for col in required_cols if col not in latencies.columns]
            if missing_cols:
                raise ValueError(f"Missing columns for pivot: {missing_cols}")
            
            # 执行透视表操作
            latency_features = latencies.pivot(
                index='window',
                columns='service_name',
                values=['duration_mean', 'duration_max', 'duration_min', 'duration_std']
            )
            
            # 重置列名
            latency_features.columns = [
                f"{service}_{metric}" 
                for metric, service in latency_features.columns
            ]
            
            # 重置索引并排序
            latency_features = latency_features.reset_index().sort_values('window')
            
            # 添加调试信息
            # print("\nFinal output shape:", latency_features.shape)
            # print("Final columns:", latency_features.columns.tolist())
            
        except Exception as e:
            print("\nDebug information:")
            print("Available columns:", latencies.columns.tolist())
            print("Data types:", latencies.dtypes)
            print("Data sample:\n", latencies.head())
            print("Error:", str(e))
            raise
        
        return latency_features