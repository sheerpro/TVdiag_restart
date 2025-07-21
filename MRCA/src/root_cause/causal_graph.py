import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from statsmodels.tsa.stattools import grangercausalitytests

class CausalGraph:
    def __init__(self, config: Dict):
        self.config = config
        self.graph = nx.DiGraph()
        
    def build_graph(self, metrics: Dict[str, np.ndarray]) -> nx.DiGraph:
        """构建因果图"""
        # 遍历所有指标对
        for service1, metric1 in metrics.items():
            for service2, metric2 in metrics.items():
                if service1 != service2:
                    # 进行Granger因果检验
                    is_causal = self._granger_causality_test(metric1, metric2)
                    
                    if is_causal:
                        self.graph.add_edge(service1, service2)
                        
        return self.graph
        
    def _granger_causality_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int = None
    ) -> bool:
        """Granger因果检验"""
        if max_lag is None:
            max_lag = self.config["causal"]["max_lag"]
            
        significance = self.config["causal"]["significance_level"]
        
        # 进行因果检验
        test_result = grangercausalitytests(
            np.column_stack([y, x]),
            maxlag=max_lag,
            verbose=False
        )
        
        # 判断是否存在因果关系
        for lag in range(1, max_lag + 1):
            p_value = test_result[lag][0]["ssr_chi2test"][1]
            if p_value < significance:
                return True
                
        return False
        
    def get_root_causes(self) -> List[str]:
        """获取根因节点"""
        # 根因节点是入度为0的节点
        return [node for node in self.graph.nodes() 
                if self.graph.in_degree(node) == 0]
    

    def get_cause_ranks(self) -> dict:
        """获取根因排名
        
        Returns:
            dict: 根因及其对应的排名分数
        """
        if not self.graph:
            return {}
            
        # 计算每个节点的入度和出度
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        # 计算排名分数（可以根据具体需求修改计算方式）
        ranks = {}
        for node in self.graph.nodes():
            # 使用出入度比率作为排名依据
            in_deg = in_degrees.get(node, 0)
            out_deg = out_degrees.get(node, 0)
            if in_deg == 0:  # 可能的根因
                ranks[node] = out_deg
                
        # 归一化分数
        max_score = max(ranks.values()) if ranks else 1
        ranks = {k: v/max_score for k, v in ranks.items()}
        
        return ranks