import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from utils import create_criteo_dataset
from model import DeepFM

import argparse
parser = argparse.ArgumentParser(description='命令行参数')
parser.add_argument('--k', type=int, help='v_dim', default=8) # 隐向量大小
parser.add_argument('--file_path', type=str, default="../Data/train.txt")
args=parser.parse_args()


if __name__ == "__main__":
    # 超参数
    batch_size = 2000
    n_epochs = 200
    learning_rate = 0.01
    embed_dim = 8

    # 读取数据集
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(args.file_path, embed_dim=embed_dim, test_size=0.2)
    # 转换为Tensor，并以DataLoader加载
    X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
    y_train, y_test = torch.from_numpy(y_train.values.reshape(-1,)), torch.from_numpy(y_test.values.reshape(-1,))
    train_dataset, test_dataset = TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)
    train_loader, test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=1024)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = DeepFM(feature_columns, feature_nums=embed_dim*len(feature_columns[1])+len(feature_columns[0]), k=30)
    model.to(device)

    # 优化器
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    # 损失函数
    criterion = nn.BCELoss()

    # 训练
    for epoch_idx in range(n_epochs):
        model.train()
        losses = []
        acc = []
        for batch_idx, (X, y) in enumerate(train_loader):
            dense_inputs, sparse_inputs = X[:, :13], X[:, 13:]
            dense_inputs, sparse_inputs, y = dense_inputs.to(device, dtype=torch.float32), sparse_inputs.to(device, dtype=torch.int32), y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            output = model(dense_inputs, sparse_inputs)
            pred = [1 if i>0.5 else 0 for i in output.cpu().detach()]
            acc.append(accuracy_score(y, pred))
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"train_loss: {sum(losses)/len(losses):5.2f} | train_acc: {sum(acc)/len(acc)*100:5.2f}%")

    # 测试
    model.eval()
    acc = []
    for batch_idx, (X, y) in enumerate(test_loader):
        dense_inputs, sparse_inputs = X[:, :13], X[:, 13:]
        dense_inputs, sparse_inputs, y = dense_inputs.to(device, dtype=torch.float32), sparse_inputs.to(device, dtype=torch.int32), y.to(device, dtype=torch.float32)
        output = model(dense_inputs, sparse_inputs)
        pred = [1 if i>0.5 else 0 for i in output.cpu().detach()]
        acc.append(accuracy_score(y, pred))

    print(f"test_acc: {sum(acc)/len(acc)*100:5.2f}%")
        