
import argparse
import os
import numpy as np
import torch
from model import EmotionRecognitionModel
from trainer import Trainer
from data_utils import load_csv_data, make_dataloader


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载邻接矩阵
    if args.adj_path.endswith('.npy'):
        A = np.load(args.adj_path)
    elif args.adj_path.endswith('.csv'):
        A = np.loadtxt(args.adj_path, delimiter=',')
    else:
        raise ValueError("Unsupported adjacency format. Please use .npy or .csv.")
    adj_matrix = torch.from_numpy(A).float().to(device)
    print(f"Adjacency matrix loaded: {A.shape}")

    
    # 2. 加载数据
    X, y = load_csv_data(args.dataset_dir, args.cols)
    print(f"Data shape: {X.shape}, Labels: {np.unique(y)}")

    # 3. 检查数据和邻接矩阵是否匹配
    if A.shape[0] != X.shape[1]:
        raise ValueError(
            f"Adjacency matrix dim ({A.shape[0]}) "
            f"does not match number of data columns ({X.shape[1]})"
        )

    # 4. 划分训练集/测试集 

    print(f"Using 7:3 train/test split (maintaining sequence order)")
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 5. 创建 Dataloaders 
    train_loader = make_dataloader(
        X_train, y_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        sequence_length=args.seq_len
    )
    test_loader = make_dataloader(
        X_test, y_test, 
        batch_size=args.batch_size, 
        shuffle=False, # 测试集不打乱
        sequence_length=args.seq_len
    )

    # 6. 初始化模型
    # input_channels 
    model = EmotionRecognitionModel(
        input_channels=1, 
        gcn_hidden=args.gcn_hidden,
        gcn_output=args.gcn_output,
        transformer_heads=args.transformer_heads,
        lstm_hidden=args.lstm_hidden,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model created with input_channels=1")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, train_loader, test_loader, optimizer, criterion, adj_matrix, device)

    # 7. 训练
    best_acc = 0.0
    for epoch in range(args.epochs):
        loss = trainer.train_one_epoch()
        acc = trainer.evaluate()
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss={loss:.4f}, Acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            # 可以在这里保存模型
            # torch.save(model.state_dict(), 'best_model.pth')
            
    print(f"\nTraining finished. Best Test Accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to EEG CSV file')
    parser.add_argument('--cols', nargs='+', help='List of EEG channels', required=True)
    parser.add_argument('--adj_path', type=str, required=True, help='Path to adjacency matrix (.npy or .csv)')
    
    # 时序参数
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length for temporal models')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # 模型架构参数
 
    parser.add_argument('--gcn_hidden', type=int, default=40)
    parser.add_argument('--gcn_output', type=int, default=40)
    parser.add_argument('--transformer_heads', type=int, default=4)
    parser.add_argument('--lstm_hidden', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=3, help='Number of emotion classes')
    
    args = parser.parse_args()

    main(args)