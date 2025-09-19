import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# 1. 准备数据
# 假设我们有 5 个节点
# x: 节点的特征矩阵，大小为 [num_nodes, num_node_features]
# 这里我们随机生成 5 个节点，每个节点有 2 个特征
x = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0],
                  [7.0, 8.0],
                  [9.0, 10.0]], dtype=torch.float)

# edge_index: 边的连接关系，大小为 [2, num_edges]
# 这是一个特殊的张量格式，第一行是源节点，第二行是目标节点
# 例如，(0, 1) 表示节点 0 连接到节点 1
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                           [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)

# y: 节点的标签，大小为 [num_nodes]
# 我们设定前 3 个节点属于类别 0，后 2 个节点属于类别 1
y = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

# 创建 PyG 的 Data 对象
# 这个对象封装了图数据的所有信息
data = Data(x=x, edge_index=edge_index, y=y)

print("--- 数据概览 ---")
print(data)
print("节点特征 x 的形状:", data.x.shape)
print("边的数量:", data.edge_index.shape[1])
print("标签 y 的形状:", data.y.shape)
print("----------------")
print("\n")

# 2. 定义 GNN 模型
# 我们使用最基本的图卷积网络（Graph Convolutional Network, GCN）
# GCNConv 聚合了邻居节点的信息
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 3. 训练模型
# 节点的特征维度是 2，最终要预测的类别数是 2
in_channels = data.num_node_features
out_channels = 2 # 两个类别
hidden_channels = 16 # 隐藏层维度可以自定义

model = GCN(in_channels, hidden_channels, out_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 为了简化，我们直接用所有节点作为训练集
train_mask = torch.ones(data.num_nodes, dtype=torch.bool)

print("--- 开始训练 ---")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    out = model(data.x, data.edge_index)

    # 计算损失
    loss = criterion(out[train_mask], data.y[train_mask])

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

print("--- 训练完成 ---")
print("\n")

# 4. 模型评估
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    # 预测结果
    pred = logits.argmax(dim=1)

    # 计算准确率
    correct = (pred == data.y).sum()
    acc = int(correct) / int(data.num_nodes)

    print("--- 预测结果 ---")
    print("预测类别:", pred.tolist())
    print("真实类别:", data.y.tolist())
    print("准确率:", f'{acc:.2f}')
    print("----------------")