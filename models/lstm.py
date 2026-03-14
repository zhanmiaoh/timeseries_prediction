import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=2, dropout=0.2, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,  # 输入特征的维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.linear = nn.Linear(hidden_size, horizon)  

    def forward(self, x):
        x, _ = self.lstm(x)  # x.shape: (batch_size, seq_length, hidden_size)
        x = x[:, -1, :] # take only the last time step's output
        x = self.linear(x)
        return x