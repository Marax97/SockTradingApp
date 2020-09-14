from datetime import datetime

import yfinance as yf
from dateutil.relativedelta import relativedelta
import ta
import utils
import pandas as pd

symbolsFile = "\\resources\\symbols.csv"
storePricesDirectory = "\\resources\\stockPricesWithIndicators\\"


def fetch_indexes_prices(years_ago):
    symbols = get_symbols_from_csv()
    years_ago = datetime.today() - relativedelta(years=years_ago)
    pd.options.display.float_format = '{:,.3f}'.format

    for symbol in symbols:
        data = yf.download(symbol, start=years_ago.strftime("%Y-%m-%d"), end=datetime.today().strftime("%Y-%m-%d"))
        data = ta.utils.dropna(data)
        data = fetch_volume_indicators(data, "High", "Low", "Close", "Volume", fillna=True)
        # data = fetch_volatility_indicators(data, "High", "Low", "Close", fillna=True)
        data = fetch_trend_indicators(data, "High", "Low", "Close", fillna=True)
        data = fetch_momentum_indicators(data, "High", "Low", "Close", "Volume", fillna=True)
        save_prices_to_csv(data.round(3), symbol)


def fetch_volume_indicators(df, high, low, close, volume, fillna=False):
    df['volume_adi'] = ta.acc_dist_index(df[high], df[low], df[close], df[volume], fillna=fillna)
    df['volume_obv'] = ta.on_balance_volume(df[close], df[volume], fillna=fillna)
    df['volume_cmf'] = ta.chaikin_money_flow(df[high], df[low], df[close], df[volume], fillna=fillna)
    df['volume_vpt'] = ta.volume_price_trend(df[close], df[volume], fillna=fillna)
    return df


def fetch_volatility_indicators(df, high, low, close, fillna=False):
    df['volatility_atr'] = ta.average_true_range(df[high], df[low], df[close], n=14, fillna=fillna)

    df['volatility_bbh'] = ta.bollinger_hband(df[close], n=20, ndev=2, fillna=fillna)
    df['volatility_bbl'] = ta.bollinger_lband(df[close], n=20, ndev=2, fillna=fillna)
    df['volatility_bbm'] = ta.bollinger_mavg(df[close], n=20, fillna=fillna)
    df['volatility_bbhi'] = ta.bollinger_hband_indicator(df[close], n=20, ndev=2, fillna=fillna)
    df['volatility_bbli'] = ta.bollinger_lband_indicator(df[close], n=20, ndev=2, fillna=fillna)

    df['volatility_kcc'] = ta.keltner_channel_central(df[high], df[low], df[close], n=10, fillna=fillna)
    df['volatility_kch'] = ta.keltner_channel_hband(df[high], df[low], df[close], n=10, fillna=fillna)
    df['volatility_kcl'] = ta.keltner_channel_lband(df[high], df[low], df[close], n=10, fillna=fillna)
    df['volatility_kchi'] = ta.keltner_channel_hband_indicator(df[high], df[low], df[close], n=10, fillna=fillna)
    df['volatility_kcli'] = ta.keltner_channel_lband_indicator(df[high], df[low], df[close], n=10, fillna=fillna)

    df['volatility_dch'] = ta.donchian_channel_hband(df[close], n=20, fillna=fillna)
    df['volatility_dcl'] = ta.donchian_channel_lband(df[close], n=20, fillna=fillna)
    df['volatility_dchi'] = ta.donchian_channel_hband_indicator(df[close], n=20, fillna=fillna)
    df['volatility_dcli'] = ta.donchian_channel_lband_indicator(df[close], n=20, fillna=fillna)

    return df


def fetch_trend_indicators(df, high, low, close, fillna=False):
    df['trend_macd'] = ta.macd(df[close], n_fast=12, n_slow=26, fillna=fillna)
    df['trend_macd_signal'] = ta.macd_signal(df[close], n_fast=12, n_slow=26, n_sign=9, fillna=fillna)
    df['trend_macd_diff'] = ta.macd_diff(df[close], n_fast=12, n_slow=26, n_sign=9, fillna=fillna)

    df['trend_adx'] = ta.adx(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend_adx_pos'] = ta.adx_pos(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend_adx_neg'] = ta.adx_neg(df[high], df[low], df[close], n=14, fillna=fillna)

    df['trend_vortex_ind_pos'] = ta.vortex_indicator_pos(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend_vortex_ind_neg'] = ta.vortex_indicator_neg(df[high], df[low], df[close], n=14, fillna=fillna)

    df['trend_vortex_diff'] = abs(
        df['trend_vortex_ind_pos'] -
        df['trend_vortex_ind_neg'])

    df['trend_cci'] = ta.cci(df[high], df[low], df[close], n=20, c=0.015, fillna=fillna)

    return df


def fetch_momentum_indicators(df, high, low, close, volume, fillna=False):
    df['momentum_mfi'] = ta.money_flow_index(df[high], df[low], df[close], df[volume], n=14, fillna=fillna)
    df['momentum_rsi'] = ta.rsi(df[close], n=14, fillna=fillna)
    df['momentum_uo'] = ta.uo(df[high], df[low], df[close], fillna=fillna)
    df['momentum_stoch'] = ta.stoch(df[high], df[low], df[close], fillna=fillna)
    df['momentum_stoch_signal'] = ta.stoch_signal(df[high], df[low], df[close], fillna=fillna)
    return df


def save_prices_to_csv(data, symbol):
    utils.create_directory_if_not_exist(storePricesDirectory)
    data.to_csv(utils.get_file_path(storePricesDirectory + symbol + '.csv'), header=True)


def get_symbols_from_csv():
    df = pd.read_csv(utils.get_file_path(symbolsFile), sep='\s*,\s*', engine='python')
    return df['Symbol'].astype(str).values.tolist()


if __name__ == "__main__":
    fetch_indexes_prices(3)
