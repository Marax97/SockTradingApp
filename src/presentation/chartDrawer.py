
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

def plot_date_prices(currentData, futureData, title):
      labels = ['History', 'True Future', 'Model Prediction']
      marker = ['.-', 'rx', 'go']

      fig, ax = plt.subplots()

      # fig.autofmt_xdate()

      pd.concat([currentData, futureData], axis=1).plot(ax=ax, rot=45)

      # pd.plot(ax=ax, legend=False, rot=45)
      # futureData.plot()
      # plt.plot(futureData, marker[1], markersize=10,
      #          label=labels[1])
      plt.show()

# plt.plot(data.Close)
# plt.plot(data.volatility_bbh, label='High BB')
# plt.plot(data.volatility_bbl, label='Low BB')
# plt.plot(data.volatility_bbm, label='EMA BB')
# plt.title('Bollinger Bands')
# plt.legend()
# plt.show()