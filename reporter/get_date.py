import wk_db
import pandas as pd

_wk_date = wk_db.read_sql('SELECT DISTINCT trade_dt FROM a_share_market ORDER BY trade_dt;', db_loc='low_freq_db')
wk_date = str(_wk_date.iloc[-1,0])
yesterday = str(_wk_date.iloc[-2,0])
# print('WK Dateset Last Date: ', wk_date)

cal = pd.read_csv('/data/linq/.qlib/qlib_data/cn_data/calendars/day.txt')
qlib_date = str(cal.iloc[-1,0])
# print('QLIB Dateset Last Date: ', qlib_date)

if qlib_date < wk_date:
    # print('New data detected')
    print(yesterday)
else:
    print('0')
