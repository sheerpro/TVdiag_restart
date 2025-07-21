import yaml
import torch
import pandas as pd
import os
import logging
from src.feature_extraction.log_feature import LogFeatureExtractor
from src.feature_extraction.trace_feature import TraceFeatureExtractor
from src.anomaly_detection.vae_detector import VAEAnomalyDetector
from src.root_cause.causal_graph import CausalGraph
from src.root_cause.rl_pruning import RLPruner
from src.evaluation.RootCauseEvaluator import RootCauseEvaluator

def load_config():
    """加载配置"""
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)
def setup_logger():
    """设置主程序日志记录器"""
    logger = logging.getLogger('MRCA')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    fh = logging.FileHandler('logs/mrca.log')
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger   

def main():
    # 设置日志记录器
    logger = setup_logger()
    logger.info("Starting MRCA...")

    # 加载配置
    config = load_config()
    
    # 1. 特征提取
    log_extractor = LogFeatureExtractor(config)
    trace_extractor = TraceFeatureExtractor(config)
    
    # 提取特征
    log_features = log_extractor.extract_templates(config["data"]["log_path"])
    trace_features = trace_extractor.extract_latency(config["data"]["trace_path"])
    
    # 合并特征
    features = pd.concat([log_features, trace_features], axis=1)

    # 在特征合并后添加数据类型检查和转换
    # print("检查特征数据类型:")
    # print(features.dtypes)

    # 转换非数值类型列
    for col in features.columns:
        if features[col].dtypes.name == 'object':
            try:
                # 尝试转换为数值类型
                features[col] = pd.to_numeric(features[col])
            except Exception as e:
                print(f"警告: 列 {col} 无法转换为数值类型")
                print(f"样本值: {features[col].head()}")
                # 可以选择删除该列或者进行其他处理
                features = features.drop(columns=[col])

    # 确保所有数据都是浮点型
    features = features.astype(float)
        
    # 计算特征维度
    config["feature_size"] = len(features.columns)

    # 2. 异常检测
    detector = VAEAnomalyDetector(config)
    
    # 划分训练集和测试集
    train_data = torch.tensor(features[:int(len(features)*0.8)].values).float()
    test_data = torch.tensor(features[int(len(features)*0.8):].values).float()
    
    # 训练VAE
    detector.train(train_data)
    
    # 检测异常
    anomalies, recon_probs = detector.detect(test_data)
    
    # 3. 根因定位
    # 获取异常时间点的指标数据
    # 3. 根因定位
    # 获取异常时间点的指标数据
    metric_data = {}
    for metric_type in config["data"]["metric_types"]:
        metric_path = os.path.join(config['data']['metric_path'], f"{metric_type}.csv")
        print(f"Reading metrics from: {metric_path}")
        
        try:
            if os.path.exists(metric_path):
                metrics = pd.read_csv(metric_path)
                # 假设指标文件包含service列来标识不同服务
                for service in config["services"]:
                    if service in metrics.columns:
                        if service not in metric_data:
                            metric_data[service] = {}
                        metric_data[service][metric_type] = metrics[service][anomalies]
            else:
                print(f"Warning: Metric file not found: {metric_path}")
        except Exception as e:
            print(f"Error reading metric file {metric_path}: {str(e)}")

    # 构建因果图
    causal_graph = CausalGraph(config)
    graph = causal_graph.build_graph(metric_data)
    
    # 使用强化学习剪枝
    pruner = RLPruner(config)
    pruned_graph = pruner.prune_graph(graph)
    
    # 获取根因
    root_causes = causal_graph.get_root_causes()
    
    print("Detected root causes:", root_causes)


     # 获取根因和排名
    root_causes = causal_graph.get_root_causes()
    root_cause_ranks = causal_graph.get_cause_ranks()
    
    logger.info(f"Detected root causes: {root_causes}")
    
    # 评估结果
    evaluator = RootCauseEvaluator(config)
    metrics = evaluator.evaluate(root_causes, root_cause_ranks)
    
    # 打印评估结果
    logger.info("\nEvaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()