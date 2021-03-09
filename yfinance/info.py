import yfinance as yf
import pandas as pd
import sys

pd.options.display.max_rows = 999
pd.options.display.max_columns = 20
pd.set_option('display.width', 1000)

argc = len(sys.argv)
if argc == 1:
    symbol = 'AAPL'
elif argc == 2:
    symbol = sys.argv[1]
else:
    print("Invalid syntax:")
    print('')
    print('python info.py <ticker symbol>')
    print('')
    exit()

ticker = yf.Ticker(symbol)
hist = ticker.history(period="max")

print('Info:')
print('=====')
print('')
for k in ticker.info.keys():
    print(f'- {k}: {ticker.info[k]}')
#print(ticker.info)
print('')

print('History:')
print('========')
print('')
print(hist.head())
print(hist.tail())
print('')

print('Actions:')
print('========')
print('')
print(ticker.actions)
print('')

print('Dividends:')
print('==========')
print('')
print(ticker.dividends)
print('')

print('Splits:')
print('=======')
print('')
print(ticker.splits)
print('')


print('Financials:')
print('===========')
print('')
print('- Financials:')
print('')
print(ticker.financials)
print('')
print('- Quarterly Financials:')
print('')
print(ticker.quarterly_financials)
print('')

print('Institutional Holders:')
print('======================')
print('')
print(ticker.institutional_holders)
print('')

print('Balance Sheet:')
print('==============')
print('')
print('- Balance Sheet:')
print('')
print(ticker.balance_sheet)
print('')
print('- Quarterly Balance Sheet')
print('')
print(ticker.quarterly_balance_sheet)
print('')

print('Cashflows:')
print('==========')
print('')
print('- Cashflow:')
print('')
print(ticker.cashflow)
print('')
print('- Quarterly Cashflow:')
print('')
print(ticker.quarterly_cashflow)
print('')

print('Earnings:')
print('=========')
print('')
print('- Earnings:')
print('')
print(ticker.earnings)
print('')
print('- Quarterly Earnings:')
print('')
print(ticker.quarterly_earnings)
print('')

print('Sustainability:')
print('===============')
print('')
print(ticker.sustainability)
print('')

print('Recommendations:')
print('===============')
print('')
print(ticker.recommendations)
print('')

print('Upcoming Calendar:')
print('==================')
print('')
print(ticker.calendar)
print('')

print('ISIN code:')
print('==========')
print('')
print(ticker.isin)
print('')

print('Options:')
print('========')
print('')
print('- Options:')
print('')
print(ticker.options)
for d in ticker.options:
    print('')
    print(f'- Option Chain for {d}')
    print('')
    opt = ticker.option_chain(d)
    print('')
    print('  - Calls:')
    print('')
    print(opt.calls)
    print('')
    print('  - Puts:')
    print('')
    print(opt.puts)
    print('')

print('')
print('Done.')