import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# ================= 数据准备 =================
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens = len(vocab), 256
device = d2l.try_gpu()
num_epochs, lr = 100, 1  # 训练轮数可调

# ================= 手写GRU实现 =================
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    W_xh, W_hh, b_h = three()
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r,
              W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    # ================= 修复 state=None 问题 =================
    if state is None:
        state = init_gru_state(inputs[0].shape[0], W_hz.shape[0], inputs[0].device)
    H, = state

    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# ================= 训练函数 =================
def train_and_record(model, train_iter, vocab, lr, num_epochs, device):
    loss = nn.CrossEntropyLoss()
    params = model.params if hasattr(model, 'params') else model.parameters()
    updater = torch.optim.SGD(params, lr=lr)
    ppl_list = []

    for epoch in range(num_epochs):
        state = None
        metric = d2l.Accumulator(2)  # [loss_sum, token_num]
        for X, Y in train_iter:
            if state is not None:
                if isinstance(state, tuple):
                    state = tuple(s.detach() for s in state)
                else:
                    state = state.detach()

            y = Y.T.reshape(-1).to(device)  # (batch*num_steps,)
            X = X.to(device)  # 不做 one-hot，交给模型内部

            y_hat, state = model(X, state)
            l = loss(y_hat, y.long())

            updater.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1)
            updater.step()

            metric.add(l * y.numel(), y.numel())

        ppl = torch.exp(metric[0] / metric[1]).item()
        ppl_list.append(ppl)

    return ppl_list

# ================= 训练手写 GRU =================
model_scratch = d2l.RNNModelScratch(
    vocab_size, num_hiddens, device, get_params,
    init_gru_state, gru
)
print("===== 训练 手写 GRU =====")
ppl_scratch = train_and_record(model_scratch, train_iter, vocab, lr, num_epochs, device)

# ================= 训练 nn.GRU =================
gru_layer = nn.GRU(vocab_size, num_hiddens)
model_builtin = d2l.RNNModel(gru_layer, len(vocab)).to(device)
print("\n===== 训练 nn.GRU =====")
ppl_builtin = train_and_record(model_builtin, train_iter, vocab, lr, num_epochs, device)

# ================= 绘制对比曲线 =================
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), ppl_scratch, label='手写 GRU')
plt.plot(range(1, num_epochs+1), ppl_builtin, label='nn.GRU')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('GRU 困惑度对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gru_compare.png")
plt.show()
