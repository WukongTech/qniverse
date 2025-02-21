import pandas as pd
import wk_db

data_root = '/data/linq/.qlib'


print('updating calendar data...')
dt = wk_db.read_sql('SELECT DISTINCT trade_dt FROM a_share_market ORDER BY trade_dt;', db_loc='low_freq_db')
dt.to_csv(f'{data_root}/qlib_data/cn_data/calendars/day.txt', index=False, header=False)

def symbol_wk2qlib(s, lower=True):
    code, mkt = s.split('.')
    if lower:
        res = mkt.lower() + code
    else:
        res = mkt.upper() + code
    return res

print('updating index data...')
symbol_wk = wk_db.read_sql('SELECT DISTINCT windcode FROM a_share_index ORDER BY windcode;', db_loc='low_freq_db')
for _, row in symbol_wk.iterrows():
    s = row['windcode']
    # print(s)
    # df = wk_db.read_sql("SELECT trade_dt, windcode, open_price, high_price, low_price, close_price, volume, amount, adj_factor FROM a_share_market WHERE windcode='{s}';".format(s=s), db_loc='low_freq_db')
    df = wk_db.read_sql("SELECT * FROM a_share_index WHERE windcode='{s}' ORDER BY trade_dt;".format(s=s), db_loc='low_freq_db')
    df = df.dropna()
    ori_raw_price = df.iloc[0]['close_price']
    ori_adj_factor = 1
    _df = pd.DataFrame()
    _df['date'] = df['trade_dt']
    _df['symbol'] = df['windcode'].apply(symbol_wk2qlib)
    _df['factor'] = 1 / (ori_raw_price*ori_adj_factor)
    _df['open'] = df['open_price'] * _df['factor']
    _df['open'] = df['open_price'] *  _df['factor']
    _df['high'] = df['high_price'] * _df['factor']   
    _df['low'] = df['low_price'] * _df['factor']
    _df['close'] = df['close_price'] *  _df['factor']
    _df['volume'] = df['volume']
    _df['amount'] = df['amount'] *  _df['factor']
    # print(df)
    # print(_df)
    wk_symbol = symbol_wk2qlib(s)
    _df.to_csv(f'{data_root}/qlib_data/cn_data/cn_1d_norm/{wk_symbol}.csv', index=False)
    # break


print('updating stock data...')
symbol_wk = wk_db.read_sql('SELECT DISTINCT windcode FROM a_share_market ORDER BY windcode;', db_loc='low_freq_db')
for _, row in symbol_wk.iterrows():
# symbol_wk = ['600000.SH']
# for row in symbol_wk:
    # s = row
    # s = row['windcode']
    # print(s)
    # df = wk_db.read_sql("SELECT trade_dt, windcode, open_price, high_price, low_price, close_price, volume, amount, adj_factor FROM a_share_market WHERE windcode='{s}';".format(s=s), db_loc='low_freq_db')
    df = wk_db.read_sql("SELECT * FROM a_share_market WHERE windcode='{s}' ORDER BY trade_dt;".format(s=s), db_loc='low_freq_db')
    ori_raw_price = df.iloc[0]['close_price']
    ori_adj_factor = df.iloc[0]['adj_factor']
    _df = pd.DataFrame()
    _df['date'] = df['trade_dt']
    _df['symbol'] = df['windcode'].apply(symbol_wk2qlib)
    _df['factor'] = df['adj_factor'] / (ori_raw_price*ori_adj_factor)
    _df['open'] = df['open_price'] * _df['factor']
    # _df['open'] = df['open_price'] *  _df['factor']
    _df['high'] = df['high_price'] * _df['factor']   
    _df['low'] = df['low_price'] * _df['factor']
    _df['close'] = df['close_price'] *  _df['factor']
    _df['volume'] = df['volume']
    _df['amount'] = df['amount'] *  _df['factor']
    # print(df)
    # print(_df)
    wk_symbol = symbol_wk2qlib(s)
    _df.to_csv(f'{data_root}/qlib_data/cn_data/cn_1d_norm/{wk_symbol}.csv', index=False)
    # break


all_code1 = wk_db.read_sql('SELECT DISTINCT windcode FROM a_share_market;', db_loc='low_freq_db')
all_code1 = wk_db.read_sql('SELECT windcode, MIN(trade_dt) as st, MAX(trade_dt) as ed FROM a_share_market GROUP BY windcode ORDER BY windcode;', db_loc='low_freq_db')
all_code1['windcode'] = all_code1['windcode'].apply(lambda s: symbol_wk2qlib(s, False))

all_code2 = wk_db.read_sql('SELECT DISTINCT windcode FROM a_share_market;', db_loc='low_freq_db')
all_code2 = wk_db.read_sql('SELECT windcode, MIN(trade_dt) as st, MAX(trade_dt) as ed FROM a_share_market GROUP BY windcode ORDER BY windcode;', db_loc='low_freq_db')
all_code2['windcode'] = all_code2['windcode'].apply(lambda s: symbol_wk2qlib(s, False))

all_code = pd.concat([all_code1, all_code2])
all_code

# %%
all_code.to_csv(f'{data_root}/qlib_data/cn_data/instruments/all.txt', index=False, header=False, sep='\t')

# %%
idx_map = {
    'csi300': "000300.SH"
}

# %%
idx_name = 'csi300'
idx_code = idx_map[idx_name]

idx_df = wk_db.read_sql(f"SELECT con_windcode, MIN(trade_dt) as st, MAX(trade_dt) as ed FROM index_components_weight WHERE index_windcode='{idx_code}' GROUP BY con_windcode ORDER BY con_windcode;", db_loc='low_freq_db')
idx_df['con_windcode'] = idx_df['con_windcode'].apply(lambda s: symbol_wk2qlib(s, False))

# %%
idx_df.to_csv(f'{data_root}/qlib_data/cn_data/instruments/{idx_name}.txt', index=False, header=False, sep='\t')




