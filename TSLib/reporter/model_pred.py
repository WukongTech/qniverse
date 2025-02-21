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
import sys
sys.path.append('/home/linq/finance/qniverse/TSLib/')
from src.dataset import MTSDatasetH

import copy
import pprint
import warnings
warnings.filterwarnings('ignore')

provider_uri = "/data/linq/.qlib/qlib_data/cn_data"  # target_dir
URI = '/home/linq/finance/qniverse/mlrun'
qlib.init(provider_uri=provider_uri, region=REG_CN)

from lilab.qlib.utils.tools import normalize_position_history, load_position_text, load_position_history, save_position_history, fill_price_position_history
from lilab.qlib.backtest.benchmark import BENCH_Step, BENCH_LPY
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--today", type=str)
# parser.add_argument("--btstart", type=str)
# parser.add_argument("--btend", type=str)
args = parser.parse_args()

TODAY = '2024-12-10'
TODAY = args.today
test_split = ('2024-01-01', TODAY)

# 4699ac22b318498998b569aa0c42eedd   5bc77afc2c434c32a2a07b5554f8a38b
pretrained_dict = {
'PatchTST': '5bc77afc2c434c32a2a07b5554f8a38b', #'b9abb5c01a4c46ecb090afdf3571eed6',
'PDF': 'e9bfe3cdf0b64966b3368feb2bd6305f',
'SegRNN': 'baea86c973bf4c06b65d038bba18a576',
'TimeBridge': 'b2c946befdd24ba980c6c01da8899a82',
'TimeMixer': 'a97f63c66ceb4596bde4b02d416df8d1',
'TimesNet': 'b850b745abfa4ebdba0b92a411786520',
'WFTNet': '47101b2cac2b4136a50562406b421986'
}

model_name = "PDF"

EXP_NAME, RID = model_name, pretrained_dict[model_name]

TOPK = 10
NDROP = 2
HT = 10

info = {
    'market': ['csi300_ext'], 
    'benchmark': ["SH000300"], 
    'feat': ["Alpha360"], 
    'label': ['r1'],
    'params': [f'topk {TOPK} ndrop {NDROP} HT {HT}']
}

filter_pipe=[]
benchmark = BENCH_Step(test_split, \
                       market=info['market'][0], \
                       benchmark=info['benchmark'][0], \
                       feat=info['feat'][0], \
                       label=info['label'][0], \
                       account=10000000, \
                       filter_pipe=filter_pipe)

dataset_info = {
    'seq_len': 20, 
    'horizon': 1, 
    'batch_size': 32, 
}

dataset = init_instance_by_config(benchmark.dataset_config)
dataset = MTSDatasetH(
    seq_len=dataset_info['seq_len'],
    horizon=dataset_info['horizon'],
    batch_size=dataset_info['batch_size'],
    handler=dataset.handler,
    segments=dataset.segments
)
# dataset.prepare('test')

with R.start(experiment_name=EXP_NAME, uri=URI):
        rid = RID

with R.start(experiment_name=EXP_NAME, uri=URI):
    recorder = R.get_recorder(recorder_id=rid)
    model = recorder.load_object("trained_model")
    # prediction
    ba_rid = R.get_recorder().id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    pred_df = recorder.load_object("pred.pkl")


rank_df = pred_df.droplevel(0).sort_values(by='score', ascending=False)
top10: pd.DataFrame = rank_df.head(10)
bottom10: pd.DataFrame = rank_df.tail(10)

# top10.to_markdown(f'./tmp/{model_name}_top10.md')
# bottom10.to_markdown(f'./tmp/{model_name}_bottom10.md')

import dataframe_image as dfi

dfi.export(top10, f'./tmp/{model_name}_top10.png',table_conversion='matplotlib')
dfi.export(bottom10, f'./tmp/{model_name}_bottom10.png',table_conversion='matplotlib')

# infodf = pd.DataFrame({'label': ['Model update date', 'Prediction generation date'],
#                        'date': ['2024-11-16', TODAY]})

# dfi.export(infodf, f'./tmp/{model_name}_info.png',table_conversion='matplotlib')