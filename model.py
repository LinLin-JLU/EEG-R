
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.3):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x 的输入形状: (B, T, N, C_in)
        B, T, N, C_in = x.shape
        
        # 1. 线性变换
        x = rearrange(x, 'b t n c -> (b t n) c')
        x = self.linear(x) # (B*T*N, C_out)
        x = rearrange(x, '(b t n) c -> b t n c', b=B, t=T, n=N) # (B, T, N, C_out)

        # 2. GCN 邻接矩阵归一化
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        D_inv_sqrt = torch.pow(adj.sum(1), -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0 # 处理孤立节点 (除以0)
        adj_norm = torch.diag(D_inv_sqrt) @ adj @ torch.diag(D_inv_sqrt) # (N, N)
        
        # 3. 图卷积: (N, N) @ (B, T, N, C_out) -> (B, T, N, C_out)
        # 使用einsum在 N 维度上进行矩阵乘法，同时保留 B, T, C 维度
        x = torch.einsum('ij,btjc->btic', adj_norm, x) # (B, T, N, C_out)
        
        return self.dropout(F.relu(x))


class GCNBlock(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.3):
        super(GCNBlock, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_feats, dropout)
        self.gcn2 = GCNLayer(hidden_feats, out_feats, dropout)
        # 残差连接必须匹配输入 C -> 输出 C
        self.residual = nn.Linear(in_feats, out_feats)

    def forward(self, x, adj):
        # x 形状: (B, T, N, C_in)
        res = self.residual(x) # (B, T, N, C_out)
        x = self.gcn1(x, adj) # (B, T, N, C_hidden)
        x = self.gcn2(x, adj) # (B, T, N, C_out)
        return x + res


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True # 关键: 确保 (Batch, Seq, Feat)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        # x 输入形状: (B, T, N, C)
        B, T, N, C = x.shape
        
        # Transformer 需要 (Batch, Seq, Feat)
        # 将 B 和 N 合并为 Batch 维度: (B*N, T, C)
        x = rearrange(x, 'b t n c -> (b n) t c')
        
        x = self.encoder(x) # 输出 (B*N, T, C)
        
        # 还原形状
        x = rearrange(x, '(b n) t c -> b t n c', b=B, n=N)
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            batch_first=True, # 关键: 确保 (Batch, Seq, Feat)
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x 输入形状: (B, T, N, C)
        
        # 得到 (B, T, C)，这代表了每个时间步的 "全图" 特征
        x = torch.mean(x, dim=2) # 形状: (B, T, C)
        
        # LSTM 输入 (Batch, Seq, Feat) -> (B, T, C)
        out, _ = self.lstm(x) # out 形状: (B, T, hidden_size * 2)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :] # 形状: (B, hidden_size * 2)
        return self.fc(out) # 形状: (B, num_classes)


class EmotionRecognitionModel(nn.Module):
    def __init__(self, input_channels=1, gcn_hidden=40, gcn_output=40,
                 transformer_heads=4, lstm_hidden=64, num_classes=3, dropout=0.5):
        super(EmotionRecognitionModel, self).__init__()
        # GCN 将 C=1 (DE特征) 映射到 C=gcn_output
        self.gcn = GCNBlock(input_channels, gcn_hidden, gcn_output, dropout)
        
        self.transformer = TransformerEncoder(gcn_output, transformer_heads)
        
        self.lstm_classifier = LSTMClassifier(gcn_output, lstm_hidden, num_classes)

    def forward(self, x, adj):
        # x 输入形状: (B, T, N) (来自 DataLoader)
        
        # 添加一个特征维度 C=1
        x = x.unsqueeze(-1) # 形状: (B, T, N, 1)
        
        # GCN 提取空间特征
        x = self.gcn(x, adj) # 形状: (B, T, N, gcn_output)
        
        # Transformer 提取时间特征
        x = self.transformer(x) # 形状: (B, T, N, gcn_output)
        
        # LSTM 分类
        return self.lstm_classifier(x) # 形状: (B, num_classes)