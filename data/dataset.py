import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class StockDataset(Dataset):

    def __init__(self, df, features, target, seq_length=60, horizon=1):
        """
        Create sequences of features and targets for time series forecasting.
        
        Args:
            df: DataFrame with datetime index
            features: List of feature column names
            target: Target column name
            seq_length: Number of past time steps to use as input
            horizon: Number of future time steps to predict (currently only supports 1)
        """

        # __init__(self, ...)：
        # 当你写 dataset = StockDataset(df, ...) 时自动执行。

        self.X = df[features].values  # 2d: (dates, features)
        self.y = df[target].values
        self.seq_length = seq_length
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.seq_length - self.horizon + 1
        # __len__(self)：
        # 当你在外部调用内置函数 len(dataset) 时，Python 会自动寻找并执行这个对象的 __len__ 方法。
        # 在这里，它计算了滑动窗口能切出多少个样本。
    
    def __getitem__(self, idx):
        x = self.X[idx : idx+self.seq_length]
        y = self.y[idx+self.seq_length : idx+self.seq_length+self.horizon]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        # __getitem__(self, idx)：
        # 当你在外部使用中括号索引 dataset[idx] 时，Python 会自动调用它。
        # 比如你写 dataset[5]，Python 就会把 5 传给 idx，
        # 然后执行里面的切片逻辑，返回第 5 个时间窗口的 (X, y) 张量。



def StockDataloader(train_dataset, val_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



# flatten sliding-window data samples for baseline evaluation
def loader_to_numpy(data_loader):
    """
    flatten sliding-window data samples for baseline evaluation
    Args:
        data_loader: a PyTorch DataLoader that yields (X_batch, y_batch) tensors
    Returns:
        X: numpy array of shape (num_samples, seq_length, num_features)
        y: numpy array of shape (num_samples, horizon)
    """

    X_all, y_all = [], []

    for X_batch, y_batch in data_loader:
        X_all.append(X_batch.numpy())
        y_all.append(y_batch.numpy())

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return X_all, y_all
