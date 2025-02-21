# %%
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict

import warnings
warnings.filterwarnings('ignore')

# %%

import argparse

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--expname", type=str)
parser.add_argument("--benchdataset", type=str)
parser.add_argument("--market", type=str)
parser.add_argument("--feat", type=str)
parser.add_argument("--label", type=str)
parser.add_argument("--topk", type=int)
parser.add_argument("--ndrop", type=int)
parser.add_argument("--ht", type=int)
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--savecsv", action="store_true")

# Parse the arguments
args = parser.parse_args()

# %%
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

# %%
from lilab.qlib.backtest.benchmark import *

EXP_NAME = args.expname

RETRAIN = args.retrain
SAVE_CSV = args.savecsv

TOPK = args.topk
NDROP = args.ndrop
HT = args.ht

info = {
    'ALGO': ['GBDT'],
    'BENCH_DATASET': [args.benchdataset],
    'market': [args.market], 
    'benchmark': ["SH000300"], 
    'feat': [args.feat], 
    'label': [args.label],
    'params': [f'topk {TOPK} ndrop {NDROP} HT {HT}']
}

# %%
model_config = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    },
}

benchmark = eval(info['BENCH_DATASET'][0])(\
                 market=info['market'][0], \
                 benchmark=info['benchmark'][0], \
                 feat=info['feat'][0], \
                 label=info['label'][0])

model = init_instance_by_config(model_config)
dataset = init_instance_by_config(benchmark.dataset_config)

strategy_config = {
    "class": "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy.signal_strategy",
    "kwargs": {
        "model": model,
        "dataset": dataset,
        "topk": TOPK,
        "n_drop": NDROP,
        "hold_thresh": HT,
    },
}

# %% [markdown]
# train model

# %%
if RETRAIN:
    with R.start(experiment_name=EXP_NAME, uri='./mlrun'):
        model.fit(dataset)
        # R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        print(rid)

# %% [markdown]
# prediction, backtest & analysis

# %%
port_analysis_config = {
    "executor": benchmark.executor_config,
    "strategy": strategy_config,
    "backtest": benchmark.backtest_config
}

# backtest and analysis
with R.start(experiment_name=EXP_NAME, uri='./mlrun'):
    recorder = R.get_recorder(recorder_id=rid)
    # model = recorder.load_object("trained_model")

    # prediction
    recorder = R.get_recorder(recorder_id=rid)
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()

# %% [markdown]
# # Reporter

# %%
from qlib.contrib.report import analysis_model, analysis_position

with R.start(experiment_name=EXP_NAME, uri='./mlrun'):
    recorder = R.get_recorder(recorder_id=rid)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ["label"]
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    recorder.save_objects(artifact_path='portfolio_analysis', **{'pred_label.pkl':pred_label})
    print(recorder)

# %% [markdown]
# ## analysis position

# %%
from lilab.qlib.utils.metrics import get_detailed_report, compute_signal_metrics

label_df = dataset.prepare("test", col_set="label")
label_df.columns = ["label"]
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

rp = get_detailed_report(info, pred_label, report_normal_df, analysis_df)
rp['EXP_NAME'] = f"'{EXP_NAME}'"
rp['rid'] = f"'{rid}'"

# %%
import os

if SAVE_CSV:
    path = f"../result/{info['BENCH_DATASET'][0]}.csv"
    if os.path.exists(path):
        rp.to_csv(path, mode='a', header=False)
    else:
        if not os.path.exists('../result'):
            os.mkdir('../result')
        rp.to_csv(path)
