import qlib
import pandas as pd
from pandas import Timestamp
import json
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data.filter import NameDFilter
from qlib.backtest.position import Position
from qlib.data import D

import copy
import pprint
import warnings
warnings.filterwarnings('ignore')

provider_uri = "/data/linq/.qlib/qlib_data/cn_data"  # target_dir
URI = '/home/linq/finance/qniverse/mlrun'
qlib.init(provider_uri=provider_uri, region=REG_CN)

from lilab.qlib.utils.tools import normalize_position_history, load_position_text, load_position_history, save_position_history, fill_price_position_history
from lilab.qlib.backtest.benchmark import BENCH_Step
import argparse
from lilab.qlib.backtest.benchmark import *


parser = argparse.ArgumentParser()
parser.add_argument("--today", type=str)
# parser.add_argument("--btstart", type=str)
# parser.add_argument("--btend", type=str)
args = parser.parse_args()

# TODAY = '2024-10-18'
# TODAY = args.today


cal = pd.read_csv('/data/linq/.qlib/qlib_data/cn_data/calendars/day.txt')
TODAY = str(cal.iloc[-1,0])
YESTODAY = str(cal.iloc[-2,0])
test_split = (TODAY, TODAY)

# EXP_NAME, rid = 'GBDT', 'afee3d7e0404433692e3f5bbbac14b99'
EXP_NAME, rid = 'GBDT', 'a6d97665eab048bea8370c04d3ed7072'

SAVE_CSV = True
TOPK = 10
# NDROP = 2
BAD_THRESH = -0.02
HT = 1
SKIP_TOPK = 70

info = {
    'ALGO': ['GBDT'],
    'BENCH_DATASET': ['BENCH_Train_Step'],
    'market': ['csi1300_ext'], 
    'benchmark': ["SH000300"], 
    'feat': ["LAlpha360"], 
    'label': ['r2'],
    'params': [f'topk {TOPK} HT {HT}']
}

# nameDFilter = NameDFilter(name_rule_re='(SH60[0-9]{4})|(SZ00[0-9]{4})')
# filter_pipe=[nameDFilter]
filter_pipe=[]
benchmark = eval(info['BENCH_DATASET'][0])(\
                 today=TODAY,
                 market=info['market'][0], \
                 benchmark=info['benchmark'][0], \
                 feat=info['feat'][0], \
                 label=info['label'][0], \
                 account=10000000, \
                 test_split=test_split,
                 filter_pipe=filter_pipe)

dataset = init_instance_by_config(benchmark.dataset_config)
# dataset.prepare('test')

with R.start(experiment_name=EXP_NAME, uri=URI):
    recorder = R.get_recorder(recorder_id=rid)
    model = recorder.load_object("trained_model")
    model.skip_topk = SKIP_TOPK
    ba_rid = R.get_recorder().id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()
    
    pred_df = recorder.load_object("pred.pkl")

rank_df = pred_df.sort_values(by='score', ascending=False)
tomorrow_return = rank_df
tomorrow_return.to_csv(f'./log/GBDTS_tomorrow_return_{TODAY}.csv')
top10: pd.DataFrame = rank_df.head(20)
bottom10: pd.DataFrame = rank_df.tail(20)

# top10.to_markdown('./tmp/gdbt_top10.md')
# bottom10.to_markdown('./tmp/gdbt_bottom10.md')

import dataframe_image as dfi

dfi.export(top10, './tmp/gdbts_top10.png',table_conversion='matplotlib')
dfi.export(bottom10, './tmp/gdbts_bottom10.png',table_conversion='matplotlib')

# infodf = pd.DataFrame({'label': ['Model update date', 'Prediction generation date'],
#                        'date': ['2024-11-16', TODAY]})

# dfi.export(infodf, './tmp/gdbt_info.png',table_conversion='matplotlib')