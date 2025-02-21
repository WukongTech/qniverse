# %%
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data.filter import NameDFilter

import warnings
warnings.filterwarnings('ignore')

# %%
provider_uri = "/data/linq/.qlib/qlib_data/cn_data"  # target_dir
URI = '/home/linq/finance/qniverse/mlrun'
qlib.init(provider_uri=provider_uri, region=REG_CN)

# %%
from lilab.qlib.backtest.benchmark import *


import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--today", type=str)
parser.add_argument("--btstart", default='2024-01-01', type=str)
parser.add_argument("--btend", type=str)
args = parser.parse_args()


# EXP_NAME, rid = 'GBDT', 'afee3d7e0404433692e3f5bbbac14b99'
EXP_NAME, rid = 'GBDT', 'a6d97665eab048bea8370c04d3ed7072'

SAVE_CSV = True
TOPK = 10
# NDROP = 2
BAD_THRESH = -0.02
HT = 1
SKIP_TOPK = 70

import dataframe_image as dfi

cal = pd.read_csv('/data/linq/.qlib/qlib_data/cn_data/calendars/day.txt')
TODAY = str(cal.iloc[-1,0])
YESTODAY = str(cal.iloc[-2,0])
infodf = pd.DataFrame({'label': ['Model update date', 'Prediction generation date', 'Top K', 'Sell thresh', 'Hold thresh'],
                    'value': ['2024-11-16', TODAY, TOPK, BAD_THRESH, HT]})

dfi.export(infodf, f'./tmp/gbdts_info.png',table_conversion='matplotlib')
test_split = (args.btstart, YESTODAY)

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

# %%
with R.start(experiment_name=EXP_NAME, uri=URI):
    recorder = R.get_recorder(recorder_id=rid)
    model = recorder.load_object("trained_model")

model.skip_topk = SKIP_TOPK
dataset = init_instance_by_config(benchmark.dataset_config)

strategy_config = {
    "class": "TopkDropoutBadStrategy",
    "module_path": "lilab.qlib.strategy.signal_strategy",
    "kwargs": {
        "model": model,
        "dataset": dataset,
        "topk": TOPK,
        "bad_thresh": BAD_THRESH,
        "hold_thresh": HT,
    },
}


EXP_NAME = 'realworld_test'

port_analysis_config = {
    "executor": benchmark.executor_config,
    "strategy": strategy_config,
    "backtest": benchmark.backtest_config
}

# backtest and analysis
with R.start(experiment_name=EXP_NAME, uri=URI):
    recorder = R.get_recorder(recorder_id=rid)
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()
    # pred_df: pd.DataFrame = recorder.load_object("pred.pkl")
    # print(pred_df)
    # pred_df.index = pred_df.index.droplevel(0)
    # sr.save(**{"pred.pkl": pred_df})

    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()


from qlib.contrib.report import analysis_model, analysis_position

with R.start(experiment_name=EXP_NAME, uri=URI):
    recorder = R.get_recorder(recorder_id=rid)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ["label"]
    # label_df.index = label_df.index.droplevel(0)
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    recorder.save_objects(artifact_path='portfolio_analysis', **{'pred_label.pkl':pred_label})
    print(recorder)

rank_df = pred_label.groupby('datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
rank_df.index = rank_df.index.droplevel(0)
rank_df.to_csv(f'./log/GBDTS_backtest_return_{TODAY}.csv')

from lilab.qlib.utils.metrics import get_detailed_report, compute_signal_metrics

# label_df = dataset.prepare("test", col_set="label")
# label_df.columns = ["label"]
# pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

rp = get_detailed_report(info, pred_label, report_normal_df, analysis_df)
rp['EXP_NAME'] = f"'{EXP_NAME}'"
rp['rid'] = f"'{rid}'"

import plotly
fig_list = analysis_position.report_graph(report_normal_df, show_notebook=False)
for i, fig in enumerate(fig_list):
    fig: plotly.graph_objs.Figure = fig
    fig.write_image(f'./tmp/gbdts_rg_{i}.jpg',engine='kaleido')

    
fig_list = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
for i, fig in enumerate(fig_list):
    fig: plotly.graph_objs.Figure = fig
    fig.write_image(f'./tmp/gbdts_rag_{i}.jpg',engine='kaleido')
    


label_df = dataset.prepare("test", col_set="label")
label_df.columns = ["label"]
# pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

fig_list = analysis_position.score_ic_graph(pred_label, show_notebook=False)

for i, fig in enumerate(fig_list):
    fig: plotly.graph_objs.Figure = fig
    fig.write_image(f'./tmp/gbdts_sig_{i}.jpg',engine='kaleido')


fig_list = analysis_model.model_performance_graph(pred_label, show_notebook=False)
for i, fig in enumerate(fig_list):
    fig: plotly.graph_objs.Figure = fig
    fig.write_image(f'./tmp/gbdts_mpg_{i}.jpg',engine='kaleido')