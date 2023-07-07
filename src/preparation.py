import numpy as np
import pandas as pd

stocks = pd.read_csv('./test_data.csv')
fields = ['open', 'close', 'low', 'high', 'volume', 'vwap']
period = 20

stock_codes = stocks['code'].unique()
for code in stock_codes:
    tmp = stocks[stocks['code'].isin([code])]
    exec("df_%s = tmp" % code[:6])

df_300750 = df_300750[fields]
df_601919 = df_601919[fields]
df_600795 = df_600795[fields]
df_300052 = df_300052[fields]
df_600522 = df_600522[fields]

y_300750 = df_300750['close'].pct_change(periods=period).dropna()
y_601919 = df_601919['close'].pct_change(periods=period).dropna()
y_600795 = df_600795['close'].pct_change(periods=period).dropna()
y_300052 = df_300052['close'].pct_change(periods=period).dropna()
y_600522 = df_600522['close'].pct_change(periods=period).dropna()
y_list = [y_300750, y_601919, y_600795, y_300052, y_600522]

df_300750 = df_300750[:-period]
df_601919 = df_601919[:-period]
df_600795 = df_600795[:-period]
df_300052 = df_300052[:-period]
df_600522 = df_600522[:-period]
stock_pool = [df_300750, df_601919, df_600795, df_300052, df_600522]  # 5x201x6


def get():
    return np.asarray(stock_pool), np.asarray(y_list)


if __name__ == '__main__':
    a, b = get()
    print(a.shape)
