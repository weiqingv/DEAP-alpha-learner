import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank(x):
    col_list = []
    for t in range(np.size(x, 1)):
        col = x[:, t]
        col = pd.Series(col.flatten()).rank().to_numpy()
        col_list.append(col)
    return np.asarray(col_list).transpose((1, 0))


def delay(x, d):
    row_list = []
    for st in range(np.size(x, 0)):
        row = x[st, :]
        row = pd.Series(row.flatten()).shift(d).fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def correlation(x, y, d):
    row_list = []
    for st in range(np.size(x, 0)):
        x_row = x[st, :]
        y_row = y[st, :]
        row = pd.Series(x_row.flatten()).rolling(d).corr(pd.Series(y_row)).fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def covariance(x, y, d):
    row_list = []
    for st in range(np.size(x, 0)):
        x_row = x[st, :]
        y_row = y[st, :]
        row = pd.Series(x_row.flatten()).rolling(d).cov(pd.Series(y_row)).fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


# def scale(x):
#     a = 1
#     data = pd.Series(x.flatten())
#     value = data.mul(a).div(np.abs(data).sum())
#     value = mean_nan(value)
#     return value


def delta(x, d):
    value = x - delay(x, d)
    return value


def ts_min(x, d):
    row_list = []
    for st in range(np.size(x, 0)):
        row = x[st, :]
        row = pd.Series(row.flatten()).rolling(d).min().fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def ts_max(x, d):
    row_list = []
    for st in range(np.size(x, 0)):
        row = x[st, :]
        row = pd.Series(row.flatten()).rolling(d).max().fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def ts_argmin(x, d):
    # value = (pd.Series(x.flatten()).rolling(d).apply(np.argmin) + 1).to_numpy()
    row_list = []
    for st in range(np.size(x, 0)):
        row = x[st, :]
        row = (pd.Series(row.flatten()).rolling(d).apply(np.argmin) + 1).fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def ts_argmax(x, d):
    row_list = []
    for st in range(np.size(x, 0)):
        row = x[st, :]
        row = (pd.Series(row.flatten()).rolling(d).apply(np.argmax) + 1).fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def ts_rank(x, d):
    row_list = []
    for st in range(np.size(x, 0)):
        row = x[st, :]
        row = pd.Series(row.flatten()).rolling(d).apply(rolling_rank).fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def rolling_rank(x):
    value = rankdata(x)[-1]
    return value


def rolling_prod(x):
    return np.prod(x)


def ts_sum(x, d):
    row_list = []
    for st in range(np.size(x, 0)):
        row = x[st, :]
        row = pd.Series(row.flatten()).rolling(d).sum().fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def ts_prod(x, d):
    row_list = []
    for st in range(np.size(x, 0)):
        row = x[st, :]
        row = pd.Series(row.flatten()).rolling(d).apply(rolling_prod).fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def ts_stddev(x, d):
    row_list = []
    for st in range(np.size(x, 0)):
        row = x[st, :]
        row = pd.Series(row.flatten()).rolling(d).std().fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def sma(x, d):
    row_list = []
    for st in range(np.size(x, 0)):
        row = x[st, :]
        row = pd.Series(row.flatten()).rolling(d).mean().fillna(method='bfill').to_numpy()
        row_list.append(row)
    return np.asarray(row_list)


def get15():
    return 10


def get5():
    return 5


def get30():
    return 30


def get60():
    return 60


def mean_nan(x):
    non_nan = x[x == x]
    x[np.isnan(x)] = np.mean(non_nan)
    return x


if __name__ == '__main__':
    x1 = [[5, 4, 3, 1, 0], [6, 7, 8, 9, 2]]
    y1 = [[1, 2], [3, 4]]
    b = np.mean(np.asarray([1, 2, 3, 4])).tolist()
    print(type(b))
