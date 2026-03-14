from sklearn.linear_model import LinearRegression
import numpy as np

def naive_forecast(X, target_index):
    """
    Use the target (eg, close) price from the last timestep
    Args:
        X: the sliding window dataset, (num_samples, seq_length/timesteps, num_features),
        target_index: the index of target feature to be preticted in features array (default: 3)
    Returns:
        (nums_patterns, 1) predictions (only one prediction target, 'target', horizon=1)
        (the shape is to match with y_test)
    """   

    return X[:, -1, target_index].reshape(-1, 1)


def moving_average_forecast(X, target_index, window=5):
    """
    Moving average forecast: uses mean of last `windows` target (eg, close) prices.
    Args:
        X: the sliding window dataset, (num_samples, seq_length/timesteps, num_features),
        target_index: the index of target feature to be preticted in features array (default: 3)
        window: the number of last target prices to average (default: 5)
    Returns:
        (nums_patterns, 1) predictions (only one prediction target, 'target', horizon=1)
    """

    if window > X.shape[1]:
        raise ValueError(f"Window size {window} is larger than sequence length {X.shape[1]}")

    return X[:, -window:, target_index].mean(axis=1).reshape(-1, 1)


def linear_regression_forecast(X_train, y_train, X_test):

    # # 步骤 A：用同样的逻辑提取训练集数据（之后把这个部分放到外面，这个函数只将X_train, y_train作为input）
    # X_train_list, y_train_list = [], []
    # for X_batch, y_batch in train_loader:
    #     X_train_list.append(X_batch.numpy())
    #     y_train_list.append(y_batch.numpy())

    # X_train = np.concatenate(X_train_list)
    # y_train = np.concatenate(y_train_list)

    # 步骤 B：将 3D 张量展平为 2D 矩阵以适配 scikit-learn
    # 形状转换: [Samples, 60, 7] -> [Samples, 420]
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 步骤 C：模型训练与预测
    lr_model = LinearRegression()
    lr_model.fit(X_train_flat, y_train)

    lr_preds = lr_model.predict(X_test_flat)

    return lr_preds