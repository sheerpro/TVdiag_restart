import os
import pandas as pd
from drain3 import TemplateMiner
from typing import Dict, List

class LogFeatureExtractor:
    def __init__(self, config: Dict):
        self.template_miner = TemplateMiner()
        self.config = config
        self.template_count = {}
        
    def extract_templates(self, log_path: str) -> pd.DataFrame:
        """从日志中提取模板并统计频率"""
        templates = []
        timestamps = []
        
        # 读取并解析日志
        with open(log_path) as f:
            for line in f:
                result = self.template_miner.add_log_message(line)
                template = result["template_mined"]
                timestamp = self._extract_timestamp(line)
                
                templates.append(template)
                timestamps.append(timestamp)
                
        # 按时间窗口统计模板频率
        df = pd.DataFrame({
            "timestamp": timestamps,
            "template": templates
        })
        
        return self._aggregate_by_window(df)
    
    def _aggregate_by_window(self, df):
        window = self.config.get("time_window", 300)  # 默认5分钟
        
        # 清理空值
        df = df.dropna(subset=["timestamp"])
        
        # 确保timestamp是数值类型
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        
        # 删除转换后的空值
        df = df.dropna(subset=["timestamp"])
        
        # 进行时间窗口聚合
        df["window"] = df["timestamp"].apply(lambda x: x // window * window)
        
        return df
        
    def _extract_timestamp(self, log_line: str) -> int:
        """从日志行提取时间戳"""
        # 根据实际日志格式实现时间戳提取
        pass