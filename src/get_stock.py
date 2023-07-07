import jqdatasdk as jq

jq.auth('18117328630', 'Simon123')
start_date = '2020-01-01'
end_date = '2020-12-01'
fields = ['open', 'close', 'low', 'high', 'volume', 'money', 'factor', 'high_limit', 'low_limit', 'avg', 'pre_close',
          'paused']
stock_list = ['600795.XSHG', '601919.XSHG', '300750.XSHE', '600522.XSHG', '300052.XSHE']
stock_panel = jq.get_price(stock_list, start_date=start_date, end_date=end_date, fq='post', fields=fields)
stock_panel['vwap'] = stock_panel['money'] / stock_panel['volume']
stock_panel.to_csv('./data.csv')
