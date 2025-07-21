import numpy as np
from typing import Dict, List, Tuple
import networkx as nx

class RLPruner:
    def __init__(self, config: Dict):
        self.config = config
        self.q_table = {}  # 状态-动作价值表
        
    def get_state(self, graph: nx.DiGraph) -> Tuple:
        """获取图的状态表示"""
        return (
            len(graph.nodes()),
            len(graph.edges()),
            max(dict(graph.degree()).values()) if graph.nodes else 0
        )
        
    def get_actions(self, graph: nx.DiGraph) -> List[str]:
        """获取可用动作"""
        return list(graph.nodes()) + ["stop"]
        
    def get_reward(self, graph: nx.DiGraph, action: str) -> float:
        """计算奖励"""
        if action == "stop":
            return self._compute_stop_reward(graph)
            
        # 移除节点的奖励
        old_score = self._evaluate_graph(graph)
        new_graph = graph.copy()
        new_graph.remove_node(action)
        new_score = self._evaluate_graph(new_graph)
        
        return new_score - old_score
        
    def _evaluate_graph(self, graph: nx.DiGraph) -> float:
        """评估图的质量"""
        # 图的复杂度惩罚
        complexity_penalty = (
            -self.config["rl"]["complexity_weight"] * 
            (len(graph.nodes()) + len(graph.edges()))
        )
        
        # 根因节点的奖励
        root_causes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        root_reward = len(root_causes) * self.config["rl"]["root_reward"]
        
        return complexity_penalty + root_reward
        
    def _compute_stop_reward(self, graph: nx.DiGraph) -> float:
        """计算停止动作的奖励"""
        score = self._evaluate_graph(graph)
        if len(graph.nodes()) < 2:  # 如果节点太少,给予惩罚
            score -= self.config["rl"]["min_nodes_penalty"]
        return score
        
    def select_action(self, state: Tuple, actions: List[str]) -> str:
        """选择动作"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in actions}
            
        # ε-贪心策略
        if np.random.random() < self.config["rl"]["epsilon"]:
            return np.random.choice(actions)
        else:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
            
    def update(
        self,
        state: Tuple,
        action: str,
        reward: float,
        next_state: Tuple
    ):
        """更新Q值"""
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.get_actions(graph)}
            
        # Q-learning更新公式
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        
        new_value = (1 - self.config["rl"]["learning_rate"]) * old_value + \
                    self.config["rl"]["learning_rate"] * (
                        reward + 
                        self.config["rl"]["discount_factor"] * next_max
                    )
                    
        self.q_table[state][action] = new_value
        
    def prune_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """使用强化学习进行图剪枝"""
        current_graph = graph.copy()
        
        while True:
            state = self.get_state(current_graph)
            actions = self.get_actions(current_graph)
            
            action = self.select_action(state, actions)
            if action == "stop":
                break
                
            reward = self.get_reward(current_graph, action)
            current_graph.remove_node(action)
            
            next_state = self.get_state(current_graph)
            self.update(state, action, reward, next_state)
            
        return current_graph