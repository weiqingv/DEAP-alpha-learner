import pandas as pd
import jqdatasdk as jq
import jqfactor_analyzer as ja
from user_functions import sma, ts_rank, ts_sum
import preparation


def alpha(vwap):
    return sma(ts_rank(ts_sum(vwap, 5), 60), 5)


start_date = '2021-01-01'
end_date = '2021-12-01'
periods = (1, 5, 20)
quantiles = 5
stock_list = ['300750.XSHE', '601919.XSHG', '600795.XSHG', '300052.XSHE', '600522.XSHG']
date_list = pd.read_csv('test_data.csv')['time'][:201].to_numpy().tolist()

stock_pool, _ = preparation.get()
data = stock_pool.transpose((2, 0, 1))[5]  # 6x5x201
factor_data = alpha(data).transpose((1, 0))  # 201x5
factor_data = pd.DataFrame(factor_data, columns=stock_list, index=pd.to_datetime(date_list))

jq.auth('18117328630', 'Simon123')
far = ja.analyze_factor(factor=factor_data,
                        weight_method='avg',
                        industry='jq_l1',
                        quantiles=quantiles,
                        periods=periods,
                        max_loss=0.25)

far.create_full_tear_sheet(
    demeaned=False, group_adjust=False, by_group=False,
    turnover_periods=None, avgretplot=(5, 15), std_bar=False
)
