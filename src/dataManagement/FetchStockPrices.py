import os
import sys
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib as mpl
from dateutil.relativedelta import relativedelta
from ta import *

symbolsFile = "./resources/symbols.csv"
storePricesDirectory = "./resources/stockPricesWithIndicators/"

def fetchStockPrices():
    symbols = getOnlySymbolsFromCsv()
    years_ago = datetime.today() - relativedelta(years=5)

    for symbol in symbols:
        data = yf.download(symbol, start=years_ago.strftime("%Y-%m-%d"), end=datetime.today().strftime("%Y-%m-%d"))
        data = utils.dropna(data)
        data = add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)
        savePricesToCSV(data, symbol)


def savePricesToCSV(data, symbol):
    data.to_csv(storePricesDirectory + symbol + '.csv', header=True)

def getOnlySymbolsFromCsv():
    df = pd.read_csv(symbolsFile, sep='\s*,\s*', engine='python')
    return df['Symbol'].astype(str).values.tolist()


if __name__ == "__main__":
    fetchStockPrices()

