import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import matplotlib.pyplot as plt

# 将音频特征加载进内存
train_data = scipy.io.loadmat('train_data.mat')
train_data = train_data['train_data']
source_graph = train_data['source_graph'][0]
target_graph = train_data['target_graph'][0]

source_graph = [i.T for i in source_graph]
target_graph = [i.T for i in target_graph]

# 为便于神经网络处理，首先将所有音频的长度全部变为10秒，不足10秒的在原音频后面添加静音片段
def pad_input(x):
    padded = [torch.tensor(i, dtype=torch.float) for i in x]
    padded = pad_sequence(padded, batch_first=True)
    padded = F.pad(padded, (0, 0, 0, 1024 - padded.shape[1]), 'constant', 0)
    return padded

n_data = len(source_graph)              # 数据长度
n_train = round(n_data * 0.8)           # 训练集长度
n_test = n_data - n_train               # 测试集长度

source_graph = pad_input(source_graph)
target_graph = pad_input(target_graph)

train_source = source_graph[:n_train]
train_target = target_graph[:n_train]
test_source = source_graph[n_train:]
test_target = target_graph[n_train:]

INPUT_SIZE = 32             # 输入特征维度
HIDDEN_SIZE = 256           # 隐藏层维度
OUTPUT_SIZE = 32            # 输出特征维度

# 编码器定义
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.encoder = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, dropout=0.5, num_layers=1)  # encoder

    def forward(self, enc_input: torch.Tensor):
        seq_len, batch_size, embedding_size = enc_input.size()
        h_0 = torch.rand(1, batch_size, self.hidden_size, device='cuda')
        c_0 = torch.rand(1, batch_size, self.hidden_size, device='cuda')
        encode_output, (encode_ht, decode_ht) = self.encoder(enc_input, (h_0, c_0))
        return encode_output, (encode_ht, decode_ht)

# 解码器定义
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.fc = nn.Linear(HIDDEN_SIZE, INPUT_SIZE)
        self.decoder = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, dropout=0.5, num_layers=1)  # encoder

    def forward(self, enc_output, dec_input):
        (h0, c0) = enc_output
        de_output, (_, _) = self.decoder(dec_input, (h0, c0))
        return de_output

# LSTM-Seq2seq整体架构定义
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.in_features = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.fc = nn.Linear(HIDDEN_SIZE, INPUT_SIZE)

    def forward(self, enc_input):
        enc_input = enc_input.permute(1, 0, 2)         # [seq_len, batch_size, embedding_size]
        # output:[seq_len, batch_size, hidden_size]
        _, (ht, ct) = self.encoder(enc_input)          # en_ht: [num_layers * num_directions, Batch_size, hidden_size]
        de_output = self.decoder((ht, ct), enc_input)  # de_output: [seq_len, batch_size, in_features]
        output = self.fc(de_output)
        output = output.permute(1, 0, 2)
        return output


# 从梅尔频谱图集合构建数据加载器
batch_size = 32
train_dataset = TensorDataset(train_source, train_target)
test_dataset = TensorDataset(test_source, test_target)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

train_loss_table = []
test_loss_table = []

# 将架构实例化
net = Net().cuda()

# 损失函数
loss_fn = nn.MSELoss()

# 定义优化器结构
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

# 开始训练流程，默认对所有训练集数据迭代10轮
for epoch in range(10):
    train_loss = 0                          # 训练集误差
    # 开始对所有数据迭代
    net.train()                             # 将模型调整为训练模式
    for source, target in train_loader:
        # 将数据加载进显存，以便使用GPU加速运算
        source = source.cuda()
        target = target.cuda()
        y_pred = net(source)
        # print(y_pred.shape, target.shape)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        loss = loss_fn(y_pred, target)      # 计算合成音频与目标音频的差异度
        train_loss += loss.item()           # 将本轮的误差累加
        loss.backward()                     # 反向传播误差值，计算偏导数
        optimizer.step()                    # 利用偏导数更新神经网络参数
    train_loss_table.append(train_loss)

    # 测试流程
    test_loss = 0                   # 测试集误差
    net.eval()                      # 将模型调整为测试模式
    with torch.no_grad():           # 关闭偏导数计算
        for source, target in test_loader:
            source = source.cuda()
            target = target.cuda()
            y_pred = net(source)
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
            target = target.reshape(target.shape[0], -1)
            loss = loss_fn(y_pred, target)
            test_loss += loss.item()
    test_loss_table.append(test_loss)
    print(f'[Epoch {epoch}]: Train Loss: {train_loss / n_train:.10f}, Test Loss: {test_loss / n_test:.10f}')

# 绘制损失函数图像
plt.figure(figsize=(10, 6))
plt.plot(train_loss_table, color='blue', label='Train Loss')
plt.plot(test_loss_table, color='orange', label='Test Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.jpg')
