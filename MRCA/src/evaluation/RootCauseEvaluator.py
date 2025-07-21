import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import os

class RootCauseEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logger()
        self.ground_truth = self._load_ground_truth()
        
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('RootCauseEvaluator')
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               'logs', 'evaluation.log')
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(fh)
    
    def _load_ground_truth(self) -> pd.DataFrame:
        """加载真实根因数据并进行预处理"""
        try:
            # 读取数据
            df = pd.read_csv(self.config["evaluation"]["ground_truth_path"])
            
            # 将时间戳转换为 Unix 时间戳
            df['st_time'] = pd.to_datetime(df['st_time'])
            df['timestamp'] = df['st_time'].astype(np.int64) // 10**9
            
            # 创建 service_metric 列
            df['service_metric'] = df.apply(lambda x: f"{x['service']}_{x['anomaly_type']}", axis=1)
            
            # 创建 is_root_cause 列 (所有记录都是根因)
            df['is_root_cause'] = 1
            
            self.logger.info(f"Loaded {len(df)} ground truth records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading ground truth: {str(e)}")
            raise
            
    def evaluate(self, detected_causes: List[str], ranks: Dict[str, int]) -> Dict[str, float]:
        """评估根因定位结果"""
        self.logger.info("Starting evaluation...")
        # 将 metrics 初始化为字典而不是列表
        metrics = dict()
        
        # 添加调试信息
        self.logger.info(f"Ground truth causes: {self.ground_truth['service_metric'].unique()}")
        self.logger.info(f"Detected causes: {detected_causes}")
        
        # 获取真实根因列表
        true_causes = self.ground_truth['service_metric'].unique()
        
        # 构建二进制标记
        y_true = []
        y_pred = []
        
        # 遍历所有可能的根因
        for cause in true_causes:
            service = cause.split('_')[0]
            detected_metrics = [m for m in detected_causes if m.startswith(service)]
            
            if detected_metrics:  # 如果检测到了这个服务的异常
                y_true.append(1)
                y_pred.append(1)
            else:  # 如果没有检测到这个服务的异常
                y_true.append(1)
                y_pred.append(0)
        
        try:
            if sum(y_true) == 0 or sum(y_pred) == 0:
                self.logger.warning("All zeros in true or predicted labels")
                metrics["precision"] = 0.0
                metrics["recall"] = 0.0
                metrics["f1"] = 0.0
            else:
                metrics["precision"] = precision_score(y_true, y_pred)
                metrics["recall"] = recall_score(y_true, y_pred)
                metrics["f1"] = f1_score(y_true, y_pred)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in metric calculation: {str(e)}")
            raise
    
    def _calculate_pr_at_k(self, detected_causes: List[str], k: int) -> float:
        """计算PR@K"""
        true_causes = self.ground_truth['service_metric'].unique()
        detected_at_k = detected_causes[:k]
        
        # 添加调试信息
        self.logger.info(f"\nCalculating PR@{k}")
        self.logger.info(f"Detected causes at k: {detected_at_k}")
        
        hits = 0
        for detected in detected_at_k:
            service = detected.split('_')[0]
            if any(true.startswith(service) for true in true_causes):
                hits += 1
                
        denominator = min(k, len(true_causes))
        if denominator == 0:
            return 0.0
        
        pr_at_k = hits / denominator
        self.logger.info(f"PR@{k}: {pr_at_k:.4f} (hits: {hits}, denominator: {denominator})")
        
        return pr_at_k
    
    def _calculate_mrr(self, ranks: Dict[str, int]) -> float:
        """计算MRR"""
        true_causes = self.ground_truth[self.ground_truth["is_root_cause"] == 1]["service_metric"].tolist()
        reciprocal_ranks = []
        
        for cause in true_causes:
            if cause in ranks:
                reciprocal_ranks.append(1.0 / (ranks[cause] + 1))
                
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0