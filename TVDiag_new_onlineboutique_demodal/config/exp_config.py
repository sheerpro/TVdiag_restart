class Config:
    def __init__(self, dataset) -> None:

        self.use_metric = False  # 是否使用指标模态
        self.use_trace = False   # 是否使用追踪模态
        self.use_log = False    # 是否使用日志模态

        # base config
        self.dataset = dataset
        self.reconstruct = False
        self.log_step = 20
        self.gpu_device = '0'

        self.modalities = ['metric', 'trace', 'log']
        
        # alert config
        self.metric_direction = True
        self.trace_op = False
        self.trace_ab_type = False

        # TVDiag modules
        self.aug_percent = 0.2
        self.aug_times = 10
        self.TO = True
        self.CM = True
        self.dynamic_weight = True

        # model config
        self.temperature = 0.3
        self.contrastive_loss_scale = 0.1
        self.batch_size = 512
        self.epochs = 500
        self.alert_embedding_dim = 128
        self.graph_hidden_dim = 64
        self.graph_out = 32
        self.graph_layers = 2
        self.linear_hidden = [64]
        self.lr = 0.001
        self.weight_decay = 0.0001

        if self.dataset == 'gaia':
            self.feat_drop = 0
            self.patience = 10
            self.ft_num = 5
            self.aggregator = 'mean'
        elif self.dataset == 'aiops22':
            if not self.trace_op:
                self.lr = 0.01
            self.feat_drop = 0.1
            self.batch_size = 128
            self.patience =20
            self.ft_num = 9
            self.aggregator = 'mean'
        elif self.dataset == 'sockshop':
            self.feat_drop = 0
            self.aug_percent = 0.4
            self.batch_size = 128
            self.patience =10
            self.ft_num = 7
            self.aggregator = 'mean'
        elif self.dataset == 'trainticket':
            self.feat_drop = 0.15                    # 高于onlineboutique，因服务间依赖更复杂
            self.batch_size = 192                    # 折衷aiops22和基础配置
            self.patience = 25                       # 延长早停等待（需更多epoch收敛）
            self.lr = 0.003                          # 低于onlineboutique
            self.ft_num = 10                         # 特征数最多（含调用链/日志/指标等）
            self.aggregator = 'mean'
            # 图结构强化配置
            self.graph_hidden_dim = 128              # 最大隐层维度（处理70+服务拓扑）
            self.graph_out = 64                      # 最大输出维度
            self.graph_layers = 4                    # 最深网络层数
            # 日志处理优化
            self.aug_percent = 0.3                   # 增强日志数据多样性

        elif self.dataset == 'onlineboutique':
            self.feat_drop = 0.1                     # 适度增加防过拟合
            self.batch_size = 256                    # 中等batch size
            self.patience = 15                       # 中等早停等待
            self.lr = 0.005                          # 中等学习率
            self.ft_num =6                          # 特征数介于aiops22和sockshop之间 应该是异常类型
            self.aggregator = 'mean'                 # 保持均值聚合
            # 针对微服务架构的优化
            self.graph_hidden_dim = 96               # 增大隐层维度以处理复杂服务关系
            self.graph_out = 48                      # 增大输出维度
            self.graph_layers = 3                    # 增加图网络层数
        else:
            raise NotImplementedError()
    
    def print_configs(self, logger):
        for attr, value in vars(self).items():
            logger.info(f"{attr}: {value}")