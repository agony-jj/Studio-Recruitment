import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. 构建你的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # --- 在这里定义你的网络层 ---

    def forward(self, x):
        # --- 在这里定义前向传播逻辑 ---
        return x

net = SimpleCNN().to(device)

# --- 请你完成这部分代码 ---
# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters())

# 4. 训练和评估模型
# 在测试集上评估模型的准确率。

# 5. 可视化
# --- Loss和Accuracy曲线 ---