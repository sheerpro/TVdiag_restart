import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class VAE(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class VAEAnomalyDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_size = config["feature_size"]
        self.model = VAE(
            input_size=input_size,
            hidden_size=config["vae"]["hidden_size"],
            latent_size=config["vae"]["latent_size"]
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["vae"]["learning_rate"]
        )
        
    def train(self, train_data: torch.Tensor):
        """训练VAE模型"""
        self.model.train()
        
        for epoch in range(self.config["vae"]["epochs"]):
            for batch in self._get_batches(train_data):
                self.optimizer.zero_grad()
                
                x = batch.to(self.device)
                recon_x, mu, log_var = self.model(x)
                
                # 重建损失
                recon_loss = F.mse_loss(recon_x, x)
                
                # KL散度
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # 总损失
                loss = recon_loss + kl_loss
                loss.backward()
                
                self.optimizer.step()
                
    def detect(self, test_data: torch.Tensor) -> torch.Tensor:
        """检测异常"""
        self.model.eval()
        
        with torch.no_grad():
            x = test_data.to(self.device)
            recon_x, mu, log_var = self.model(x)
            
            # 计算重建概率
            recon_prob = self._reconstruction_probability(x, recon_x, mu, log_var)
            
            # 根据阈值判断异常
            threshold = self.config["vae"]["reconstruction_threshold"]
            anomalies = recon_prob < threshold
            
            return anomalies, recon_prob
            
    def _reconstruction_probability(
    self,
    x: torch.Tensor,
    recon_x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor
    ) -> torch.Tensor:
        """计算重建概率"""
        # 计算重建误差
        recon_error = F.mse_loss(recon_x, x, reduction='none')  # [batch_size, feature_size]
        
        # 确保维度匹配
        var = torch.exp(log_var)  # [batch_size, latent_size]
        var = var.unsqueeze(-1).expand(-1, -1, x.size(-1))  # [batch_size, latent_size, feature_size]
        
        # 计算概率密度
        prob = torch.mean(
            (1.0 / torch.sqrt(2*torch.pi*var)) * torch.exp(-recon_error.unsqueeze(1)/(2*var)),
            dim=1  # 在潜在维度上取平均
        )
        
        return prob.mean(dim=1)  # 在特征维度上取平均
        
    def _get_batches(self, data: torch.Tensor):
        """生成批次数据"""
        batch_size = self.config["vae"]["batch_size"]
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]