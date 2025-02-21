# %%
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data.filter import NameDFilter


import universal as up
from universal import tools, algos
from universal.algos import *

from lilab.qlib.model.meanvar_signal import SimpleMeanVar
from lilab.qlib.utils.tools import get_return_dataframe
from lilab.qlib.strategy.meanvar_strategy import MeanVarStrategy

from upadapter import UPAdapter

import warnings
warnings.filterwarnings('ignore')

provider_uri = "/data/linq/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from lilab.qlib.backtest.benchmark import *
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--today", type=str)
parser.add_argument("--btstart", type=str)
parser.add_argument("--btend", type=str)
args = parser.parse_args()

ALGO = 'BNN'
EXP_NAME, URI = ALGO, '/home/linq/finance/qniverse/mlrun'

test_split = (args.btstart, args.btend)
algo = BNN()
up_adapter = UPAdapter('csi300_ext', test_split[0], test_split[1], algo)

info = {
    'ALGO': [ALGO],
    'BENCH_DATASET': ['BENCH_Step'],
    'market': ['csi300_ext'], 
    'benchmark': ["SH000300"], 
    'feat': ["Alpha360"], 
    'label': ['r1'],
    'params': ['']
}

# nameDFilter = NameDFilter(name_rule_re='(SH60[0-9]{4})|(SZ00[0-9]{4})')
# filter_pipe=[nameDFilter]
filter_pipe=[]
benchmark = eval(info['BENCH_DATASET'][0])(\
                 market=info['market'][0], \
                 benchmark=info['benchmark'][0], \
                 feat=info['feat'][0], \
                 label=info['label'][0], \
                 account=10000000, \
                 test_split=test_split,
                 filter_pipe=filter_pipe)

dataset = init_instance_by_config(benchmark.dataset_config)

# %%
df = up_adapter.data
up_adapter.predict()

# %%
df = up_adapter.get_weight()
pos_df = up_adapter.get_qlib_weight()

# %%
strategy_config = {
    "class": "WeightCopyStrategy",
    "module_path": "lilab.qlib.strategy.signal_strategy",
    "kwargs": {
        "risk_degree": 1.0,
        "pos_df": pos_df
    },
}

EXP_NAME = 'realworld_test'

port_analysis_config = {
    "executor": benchmark.executor_config,
    "strategy": strategy_config,
    "backtest": benchmark.backtest_config
}

model = SimpleMeanVar(df, '2023-12-31')

# backtest and analysis
with R.start(experiment_name=EXP_NAME, uri=URI):
    recorder = R.get_recorder()
    rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()

# %%
from qlib.contrib.report import analysis_model, analysis_position

with R.start(experiment_name=EXP_NAME, uri=URI):
    recorder = R.get_recorder(recorder_id=rid)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    print(recorder)


import plotly
fig_list = analysis_position.report_graph(report_normal_df, show_notebook=False)
for i, fig in enumerate(fig_list):
    fig: plotly.graph_objs.Figure = fig
    fig.write_image(f'./tmp/{ALGO}_rg_{i}.jpg',engine='kaleido')

    
fig_list = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
for i, fig in enumerate(fig_list):
    fig: plotly.graph_objs.Figure = fig
    fig.write_image(f'./tmp/{ALGO}_rag_{i}.jpg',engine='kaleido')
    

infodf = pd.DataFrame({'label': ['Model update date', 'Prediction generation date'],
                       'date': ['-', args.btend]})

import dataframe_image as dfi
dfi.export(infodf, f'./tmp/{ALGO}_info.png',table_conversion='matplotlib')

