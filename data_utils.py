
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_csv_data(file_path, cols):
    """
    加载CSV数据。
    """
    try:
        df = pd.read_csv(file_path, encoding='CP949')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path) # 尝试使用默认 UTF-8
    
    # 确保列存在
    valid_cols = [col for col in cols if col in df.columns]
    missing_cols = set(cols) - set(valid_cols)
    if missing_cols:
        print(f"警告: 找不到以下列，将跳过: {missing_cols}")
        
    X = df[valid_cols].values
    y = df['label'].values if 'label' in df.columns else df.iloc[:, -1].values
    return X, y

class EEGSequenceDataset(Dataset):
    """
    一个自定义的Dataset类，用于从连续的EEG数据中创建序列。
    """
    def __init__(self, X, y, sequence_length):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        # 可生成的序列总数
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        # 序列的起始索引是 idx
        # 序列的结束索引是 idx + self.sequence_length
        X_sequence = self.X[idx : idx + self.sequence_length]
        
        # 我们使用序列中 *最后一个* 时间点的标签作为整个序列的标签
        y_label = self.y[idx + self.sequence_length - 1]
        
        return torch.tensor(X_sequence, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)

def make_dataloader(X, y, batch_size=32, shuffle=True, normalization=True, sequence_length=10):
    """
    创建Dataloader，现在支持序列化数据。
    """
    if normalization:
        # 注意：在K-Fold交叉验证中，scaler应该在训练集上fit，然后在测试集上transform。
        # 为简单起见，这里我们在当前数据块上fit_transform。
        scaler = StandardScaler()
        X = scaler.fit_transform(X) # X 是 (TotalTimeSteps, NumChannels)

    # 使用自定义Dataset创建序列
    dataset = EEGSequenceDataset(X, y, sequence_length)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader