import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def engineer_features(df, new_features=["return", "log_return"]):
    """
    Engineer features from raw stock data.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with engineered features
    """

    for feature in new_features:
        if feature == "return":
            df["return"] = df['Close'].pct_change()  # 计算日收益率
            # the first row becomes NaN

        elif feature == "log_return":
            df["log_return"] = np.log(df['Close'] / df['Close'].shift(1))
            # after df['Close'].shift(1)), the first row becomes NaN
    
    # 之后还可以加入别的features

    df = df.dropna()

    return df


def create_split(df, train_end="2020-12-31", val_end="2022-12-31"):
    """
    Create time-aware train/validation/test splits.
    
    Args:
        df: DataFrame with datetime index
        train_end: End date for training set
        val_end: End date for validation set
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # 之前做的是：先 train-test split for time series
    # 再scale（下一个函数）
    # 再create sliding-window (sequences) dataset for training and test each （不在prepare_data当中）
    # 这个函数只做第一步

    train_df = df.loc[:train_end].copy()  # loc: 按index索引，前面已经把index设为Date了
    val_df = df.loc[train_end:val_end].copy()
    val_df = val_df.iloc[1:]  # Remove overlap (loc索引 左右闭区间)
    test_df = df.loc[val_end:].copy()
    test_df = test_df.iloc[1:]  # Remove overlap
    
    return train_df, val_df, test_df


def scale_features(train_df, val_df, test_df, features, target):

    scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_df_scaled = train_df.copy()
    train_df_scaled[features] = scaler.fit_transform(train_df[features]) # (num_samples, num_features)
    train_df_scaled[target] = target_scaler.fit_transform(train_df[[target]])

    val_df_scaled = val_df.copy()
    val_df_scaled[features] = scaler.transform(val_df[features])
    val_df_scaled[target] = target_scaler.transform(val_df[[target]])

    test_df_scaled = test_df.copy()
    test_df_scaled[features] = scaler.transform(test_df[features])
    test_df_scaled[target] = target_scaler.transform(test_df[[target]])

    # create a separate scaler for target='Close'

    return scaler, target_scaler, train_df_scaled, val_df_scaled, test_df_scaled


def target_inverse_scale(y_test, prediction_dict, target_scaler):
    """
    inverse_transform the scaled target values back to original scale.
    Args:
        y_test, numpy array of shape (num_samples, horizon)
        prediction_dict = {
        'Naive Baseline': naive_preds,
        'Moving Average': ma_preds,
        'Linear Regression': lr_preds,
        'LSTM': lstm_preds,
        'Transformer': transformer_preds
        }
        each is a numpy array of shape (num_samples, horizon)
    Return:
        original scale of y_test, shape (num_samples, horizon)
        a dict of preds from different models, each shape (num_samples, horizon)
    """

    original_shape = y_test.shape
    y_test = y_test.reshape(-1, 1)  # inverse_transform requires (num_samples, num_features=1) 
    y_test_raw = target_scaler.inverse_transform(y_test)
    y_test_raw = y_test_raw.reshape(original_shape)  # reshape back to (num_samples, horizon)

    prediction_dict_raw = {}
    for label, preds in prediction_dict.items():
        original_shape = preds.shape
        preds = preds.reshape(-1, 1)
        prediction_dict_raw[label] = target_scaler.inverse_transform(preds).reshape(original_shape)

    return y_test_raw, prediction_dict_raw


# def prepare_data(df, train_end, val_end, features, target):

#     # Create splits
#     train_df, val_df, test_df = create_split(df, train_end, val_end)
#     # print("\nAfter creating train/val/test splits:")
#     # print("Train:", train_df.shape, train_df.index.min(), train_df.index.max())
#     # print("Val:", val_df.shape, val_df.index.min(), val_df.index.max())
#     # print("Test:", test_df.shape, test_df.index.min(), test_df.index.max())
    
#     # scale features
#     scaler, target_scaler, train_df_scaled, val_df_scaled, test_df_scaled = scale_features(train_df, val_df, test_df, features, target)
    
#     return {
#         'train_df': train_df_scaled,
#         'val_df': val_df_scaled,
#         'test_df': test_df_scaled,
#         'train_df_raw': train_df,
#         'val_df_raw': val_df,
#         'test_df_raw': test_df,
#         'scaler': scaler,
#         'target_scaler': target_scaler
#     }
