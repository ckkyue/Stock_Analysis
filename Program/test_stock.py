# Imports
import datetime as dt
from helper_functions import generate_end_dates, stock_market
import pandas as pd
from plot import *
from stock_screener import EM_rating, stoploss_target
from technicals import *

# Start of the program
start = dt.datetime.now()

# Initial setup
current_date = start.strftime("%Y-%m-%d")

# Choose the stocks
stocks = ["0700.HK", "1810.HK", "1339.HK", "2618.HK", "3690.HK", "6881.HK"]

# Iterate over stocks
for stock in stocks:
    df = get_df(stock, current_date)
    plot_close(stock, df, show=200, save=True)
    plot_MFI_RSI(stock, df, show=375, save=True)
    plot_stocks(["^GSPC", "^GSPC", stock], current_date, save=True)

# Get the stop loss and target price of a stock
for stock in stocks:
    df = get_df(stock, current_date)
    current_close = df["Close"].iloc[-1]
    stoploss, stoploss_pct, target, target_pct = stoploss_target(stock, current_close, current_date)
    print(f"Current close: {round(current_close, 2)}.")
    print(f"Stoploss: {stoploss}, {stoploss_pct} (%).")
    print(f"Target price: {target}, {target_pct} (%).")

# Print the end time and total runtime
end = dt.datetime.now()
print(end, "\n")
print("The program used", end - start)