import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from tsai.models.ResNet import ResNet
# from FCN import FCN
# import EarlyStopping
from pytorchtools import EarlyStopping

from preprocess import bandpass_filter

input_dir = './slice_data'
bcgData = []
sum = 0
labels = []
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    data = np.array(pd.read_csv(file_path, header=None), dtype=np.float32).squeeze()
    data = bandpass_filter(data, 0.7, 5, sampling_rate=100)
    bcgData.append(data)
    if file_name[0] == 'H':
        labels.append(1)
    elif file_name[0] == 'N':
        labels.append(0)
    sum += 1
    if sum % 1000 == 0:
        print(sum)
    # print(data.shape)

print('data read complete!')
bcgData = np.array(bcgData, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(bcgData, labels, test_size=0.3, random_state=42)

# print(bcgData.shape)
# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转为Pytorch张量
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 检查GPU设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')

# 创建模型实例并移动到GPU
num_classed = len(torch.unique(y_train))
# print(num_classed)
model = ResNet(1, num_classed).to(device)
# print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epoches = 100
train_losses = []
val_losses = []
early_stopping = EarlyStopping(patience=7, verbose=False)
for epoch in range(num_epoches):
    model.train()

    # running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for X_batch, y_batch in train_bar:
        optimizer.zero_grad()
        # 移动数据到GPU
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)

        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()

    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        val_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    print(f'Epoch {epoch + 1}/{num_epoches}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
model.load_state_dict(torch.load('checkpoint.pt'))
# torch.save(model.state_dict(), './save_models/fcn_state_hpt_100.pth')
# torch.save(model, './save_models/fcn_all_hpt_100.pth')
print('模型保存完毕！')
# 评估模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        _, predicted = torch.max(output, dim=1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
print('correct:', correct)
print('total:', total)
print(f'train Accuracy: {correct / total}')

correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        _, predicted = torch.max(output, dim=1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
print('correct:', correct)
print('total:', total)
print(f'test Accuracy: {correct / total}')
