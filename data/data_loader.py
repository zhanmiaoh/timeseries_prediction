import akshare as ak
import pandas as pd
import numpy as np

def load_stock_data(ticker, start_date, end_date):
    """
    load data from akshare, reframing,
    and return a pandas DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
    """

    # 1. 从 AkShare 获取数据 (adjust="" 对应 yfinance 的 auto_adjust=False，即获取不复权的原始数据)
    try: 
        df = ak.stock_us_daily(symbol=ticker, adjust="")
    except Exception as e:
        raise RuntimeError(f"Failed to download data for {ticker} using akshare: {e}")

    if df is None or len(df) is None or df.empty:
        raise RuntimeError(f"Failed to download data for {ticker}: empty result")

    # 2. 重命名列以匹配 yfinance 的大写格式
    rename_map = {
        'date': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    df = df.rename(columns=rename_map)

    # 3. 转换数据类型：确保 Date 列是 datetime 格式，并将其设为索引
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # 确保 OHLCV 数据是数值型（AkShare 有时返回的数据类型可能是 object）
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. 根据 start_date 和 end_date 进行本地切片
    # 确保索引是排序好的，否则切片会报错
    df = df.sort_index() 
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df.loc[start_dt:end_dt]

    return df