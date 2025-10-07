import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt

# 从kaggle中下载数据集
dataset_path = kagglehub.dataset_download("lakshmi25npathi/bike-sharing-dataset")

# 加载数据集
orignal_data_day = pd.read_csv(os.path.join(dataset_path,'day.csv'))

#得到输入temp和输出cnt
#这里的inputs，outputs对应X_df, Y_df
inputs, outputs = orignal_data_day['temp'], orignal_data_day['cnt']

#转化为张量
#这里的X,Y对应X_tensor, Y_tensor
X, Y = torch.tensor(inputs.values,dtype=torch.float32), torch.tensor(outputs.values,dtype=torch.float32)
X, Y = X.reshape(-1,1), Y.reshape(-1,1)
#定义线性回归神经网络
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features:int, out_features:int,bias = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias = bias)

    def forward(self,x):
        return self.linear(x)
    
#定义一个简单的MLP
class NonLinearModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_features = 10, bias = True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

#定义初始化方法
def init_params(m):
    if type(m) == nn.Module:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.fill_(0.)

#创建网络并初始化
#net = LinearRegressionModel(1,1)
net = NonLinearModel(1,1,hidden_features=10)
net.apply(init_params)
#定义损失函数
criterion = nn.MSELoss(reduction='mean')

#加载数据集并设置超参数
lr = 0.1
epoch_nums = 500
batch_size = 32
# 数据标准化（重要！）
X_mean, X_std = X.mean(), X.std()
X_normalized = (X - X_mean) / X_std

Y_mean, Y_std = Y.mean(), Y.std()
Y_normalized = (Y - Y_mean) / Y_std
dataset = TensorDataset(X_normalized,Y_normalized)
train_data = DataLoader(dataset,batch_size=batch_size ,shuffle=True)
#选择随机梯度下降作为优化器
optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=1e-5)
loss_data = []# 由于存储损失值

# 开始训练
for epoch in range(epoch_nums + 1):
    epoch_losses = []  # 存储当前epoch的所有batch损失
    for x, y in train_data:
        optimizer.zero_grad()
        y_hat = net(x)
        loss = criterion(y_hat, y)  # 注意参数顺序：预测值在前，真实值在后
        # loss是一个零维张量
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())  # 存储标量值
    
    # 计算当前epoch的平均损失
    avg_loss = np.mean(epoch_losses)
    loss_data.append(avg_loss)  # 存储每个epoch的平均损失
    
    if epoch % 10 == 0 and epoch != 0:
        print(f'Epoch:{epoch} Loss: {avg_loss:.4f}')

# 用训练好的模型进行预测得到最终的拟合直线
with torch.no_grad():
    y_pred = net(X)

# 打印训练后得到的参数
print("训练后的参数:")
print(net.state_dict())

# 创建图表
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 显示初始数据的散点图和拟合曲线
axs[0].scatter(X.numpy(), Y.numpy(), s=10, alpha=0.6)# 设置alpha使点变为半透明的
axs[0].set_xlabel('temp')
axs[0].set_ylabel('cnt')
axs[0].set_title('original distribution & final fitted curve')

# 绘制拟合曲线
axs[0].plot(X.numpy(), y_pred.detach().numpy(), color='r', linewidth=2)

# 绘制epoch-loss图
axs[1].plot(range(len(loss_data)), loss_data, color='b')
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('loss')
axs[1].set_title('epoch-loss')
axs[1].grid(True) #使用网格

plt.tight_layout()
plt.show()