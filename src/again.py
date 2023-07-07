import numpy as np
import pandas as pd

# x1 = pd.Series([0.3, 0.4])
# y1 = pd.Series([0.7, 0.6])
# n = x1.count()
# x1.index = np.arange(n)
# y1.index = np.arange(n)
#
# d = (x1.sort_values().index - y1.sort_values().index) ** 2
# dd = d.to_series().sum()
#
# p = 1 - 6 * dd / (n * (n ** 2 - 1))
#
# print(p)
# print(y1.corr(x1, method='spearman'))

fields = ['open', 'high', 'low', 'avg', 'pre_close', 'high_limit', 'low_limit', 'close']
stock_price = pd.read_csv('./test.csv')
source = stock_price[fields].values
source = source[:-1]
target = stock_price['pct'].values
target = target[1:]

stock_list = np.split(source, 7)  # 7x14x8
stock_list = np.asarray(stock_list).transpose((1, 0, 2))  # 14x7x8

rank_ic_list = []
for t in stock_list:
    expo_list = []
    for stock in t:
        expo = func(*stock)
        expo_list.append(expo)
    # calc y
    rank_ic = 1
    rank_ic_list.append(rank_ic)

