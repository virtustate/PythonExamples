import yfinance as yf

# from yahoo_finance import Share
#
# yahoo = Share('YHOO')
# print(yahoo.get_open())
# print(yahoo.get_price())
# print(yahoo.get_trade_datetime())
# exit

# msft = yf.Ticker("MSFT")
# print(msft)
"""
returns
<yfinance.Ticker object at 0x1a1715e898>
"""

# get stock info
# print(msft.info)

"""
returns:
{
 'quoteType': 'EQUITY',
 'quoteSourceName': 'Nasdaq Real Time Price',
 'currency': 'USD',
 'shortName': 'Microsoft Corporation',
 'exchangeTimezoneName': 'America/New_York',
  ...
 'symbol': 'MSFT'
}
"""

# print(msft.history(period='1d'))

# get historical market data, here max is 5 years.
# print(msft.history(period="max"))

data = yf.download(
    # or pdr.get_data_yahoo(...
    # tickers list or string as well
    tickers="SPY AAPL MSFT",

    # use "period" instead of start/end
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # (optional, default is '1mo')
    # period = "1d",
    start='2020-05-11',
    end='2020-05-12',

    # fetch data by interval (including intraday if period < 60 days)
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # (optional, default is '1d')
    interval="1m",

    # group by ticker (to access via data['SPY'])
    # (optional, default is 'column')
    group_by='ticker',

    # adjust all OHLC automatically
    # (optional, default is False)
    auto_adjust=True,

    # download pre/post regular market hours data
    # (optional, default is False)
    prepost=True,

    # use threads for mass downloading? (True/False/Integer)
    # (optional, default is True)
    threads=True,

    # proxy URL scheme use use when downloading?
    # (optional, default is None)
    proxy=None
)

print(data)
