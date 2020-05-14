from datetime import datetime

import yfinance as yf
from dateutil.relativedelta import relativedelta
from ta import *
from utils import get_file_path

symbolsFile = "\\resources\\symbols.csv"
storePricesDirectory = "\\resources\\stockPricesWithIndicators\\"


def fetch_indexes_prices(years_ago):
    symbols = get_symbols_from_csv()
    years_ago = datetime.today() - relativedelta(years=years_ago)
    pd.options.display.float_format = '{:,.3f}'.format

    for symbol in symbols:
        data = yf.download(symbol, start=years_ago.strftime("%Y-%m-%d"), end=datetime.today().strftime("%Y-%m-%d"))
        data = utils.dropna(data)
        data = fetch_volume_indicators(data, "High", "Low", "Close", "Volume", fillna=True)
        data = fetch_volatility_indicators(data, "High", "Low", "Close", fillna=True)
        data = fetch_trend_indicators(data, "High", "Low", "Close", fillna=True)
        data = fetch_momentum_indicators(data, "High", "Low", "Close", "Volume", fillna=True)
        save_prices_to_csv(data.round(3), symbol)


def fetch_volume_indicators(df, high, low, close, volume, fillna=False):
    df['volume_adi'] = acc_dist_index(df[high], df[low], df[close], df[volume], fillna=fillna)
    df['volume_obv'] = on_balance_volume(df[close], df[volume], fillna=fillna)
    df['volume_cmf'] = chaikin_money_flow(df[high], df[low], df[close], df[volume], fillna=fillna)
    df['volume_vpt'] = volume_price_trend(df[close], df[volume], fillna=fillna)
    return df


def fetch_volatility_indicators(df, high, low, close, fillna=False):
    df['volatility_atr'] = average_true_range(df[high], df[low], df[close], n=14, fillna=fillna)

    df['volatility_bbh'] = bollinger_hband(df[close], n=20, ndev=2, fillna=fillna)
    df['volatility_bbl'] = bollinger_lband(df[close], n=20, ndev=2, fillna=fillna)
    df['volatility_bbm'] = bollinger_mavg(df[close], n=20, fillna=fillna)
    df['volatility_bbhi'] = bollinger_hband_indicator(df[close], n=20, ndev=2, fillna=fillna)
    df['volatility_bbli'] = bollinger_lband_indicator(df[close], n=20, ndev=2, fillna=fillna)

    df['volatility_kcc'] = keltner_channel_central(df[high], df[low], df[close], n=10, fillna=fillna)
    df['volatility_kch'] = keltner_channel_hband(df[high], df[low], df[close], n=10, fillna=fillna)
    df['volatility_kcl'] = keltner_channel_lband(df[high], df[low], df[close], n=10, fillna=fillna)
    df['volatility_kchi'] = keltner_channel_hband_indicator(df[high], df[low], df[close], n=10, fillna=fillna)
    df['volatility_kcli'] = keltner_channel_lband_indicator(df[high], df[low], df[close], n=10, fillna=fillna)

    df['volatility_dch'] = donchian_channel_hband(df[close], n=20, fillna=fillna)
    df['volatility_dcl'] = donchian_channel_lband(df[close], n=20, fillna=fillna)
    df['volatility_dchi'] = donchian_channel_hband_indicator(df[close], n=20, fillna=fillna)
    df['volatility_dcli'] = donchian_channel_lband_indicator(df[close], n=20, fillna=fillna)

    return df


def fetch_trend_indicators(df, high, low, close, fillna=False):
    df['trend_macd'] = macd(df[close], n_fast=12, n_slow=26, fillna=fillna)
    df['trend_macd_signal'] = macd_signal(df[close], n_fast=12, n_slow=26, n_sign=9, fillna=fillna)
    df['trend_macd_diff'] = macd_diff(df[close], n_fast=12, n_slow=26, n_sign=9, fillna=fillna)

    df['trend_adx'] = adx(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend_adx_pos'] = adx_pos(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend_adx_neg'] = adx_neg(df[high], df[low], df[close], n=14, fillna=fillna)

    df['trend_vortex_ind_pos'] = vortex_indicator_pos(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend_vortex_ind_neg'] = vortex_indicator_neg(df[high], df[low], df[close], n=14, fillna=fillna)

    df['trend_vortex_diff'] = abs(
        df['trend_vortex_ind_pos'] -
        df['trend_vortex_ind_neg'])

    df['trend_cci'] = cci(df[high], df[low], df[close], n=20, c=0.015, fillna=fillna)

    return df


def fetch_momentum_indicators(df, high, low, close, volume, fillna=False):
    df['momentum_mfi'] = money_flow_index(df[high], df[low], df[close], df[volume], n=14, fillna=fillna)
    df['momentum_rsi'] = rsi(df[close], n=14, fillna=fillna)
    df['momentum_uo'] = uo(df[high], df[low], df[close], fillna=fillna)
    df['momentum_stoch'] = stoch(df[high], df[low], df[close], fillna=fillna)
    df['momentum_stoch_signal'] = stoch_signal(df[high], df[low], df[close], fillna=fillna)
    return df


def save_prices_to_csv(data, symbol):
    data.to_csv(get_file_path(storePricesDirectory + symbol + '.csv'), header=True)


def get_symbols_from_csv():
    df = pd.read_csv(get_file_path(symbolsFile), sep='\s*,\s*', engine='python')
    return df['Symbol'].astype(str).values.tolist()


if __name__ == "__main__":
    fetch_indexes_prices(5)
