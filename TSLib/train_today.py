import argparse

import qlib
import ruamel.yaml as yaml
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from src.dataset import MTSDatasetH
from lilab.qlib.backtest.benchmark import *
import plotly
import torch
import warnings
warnings.filterwarnings('ignore')
import os,sys
os.chdir(sys.path[0])

# python train_today.py --config_file configs/config_test.yaml

pretrained_dict = {
    'PatchTST': '5bc77afc2c434c32a2a07b5554f8a38b', #'b9abb5c01a4c46ecb090afdf3571eed6',
    'PDF': 'e9bfe3cdf0b64966b3368feb2bd6305f',
    'SegRNN': 'baea86c973bf4c06b65d038bba18a576',
    'TimeBridge': 'b2c946befdd24ba980c6c01da8899a82',
    'TimeMixer': 'a97f63c66ceb4596bde4b02d416df8d1',
    'TimesNet': 'b850b745abfa4ebdba0b92a411786520',
    'WFTNet': '47101b2cac2b4136a50562406b421986'
}

def main(seed, config_file="configs/config_wftnet.yaml", btstart="2024-01-01"):

    # set random seed
    with open(config_file) as f:
        config = yaml.safe_load(f)

    model_name = config["task"]["model"]["kwargs"]["model_type"]
    EXP_NAME, RID = model_name, pretrained_dict[model_name]

    # seed_suffix = "/seed1000" if "init" in config_file else f"/seed{seed}"
    seed_suffix = ""
    config["task"]["model"]["kwargs"].update(
        {"seed": seed, "logdir": config["task"]["model"]["kwargs"]["logdir"] + seed_suffix}
    )

    # initialize workflow
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        region=config["qlib_init"]["region"],

    )

    # TODAY = today
    cal = pd.read_csv('/data/linq/.qlib/qlib_data/cn_data/calendars/day.txt')
    TODAY = str(cal.iloc[-1,0])
    YESTERDAY = str(cal.iloc[-2,0])
    pred_split = (btstart, YESTERDAY)
    yield_split = (TODAY, TODAY)

    # pred
    pred_benchmark = eval(config["task"]["qlib_dataset"]["class"])(**config["task"]["qlib_dataset"]["kwargs"], test_split=pred_split)
    pred_dataset = init_instance_by_config(pred_benchmark.dataset_config)
    config["task"]["dataset"]["kwargs"]["handler"] = pred_dataset.handler
    config["task"]["dataset"]["kwargs"]["segments"] = pred_dataset.segments
    print(pred_dataset.segments)
    # origin data -> MTS DataLoader
    pred_dataset = eval(config["task"]["dataset"]["class"])(**config["task"]["dataset"]["kwargs"])

    # yield
    yield_benchmark = eval(config["task"]["qlib_dataset"]["class"])(**config["task"]["qlib_dataset"]["kwargs"], test_split=yield_split)
    yield_dataset = init_instance_by_config(yield_benchmark.dataset_config)
    config["task"]["dataset"]["kwargs"]["handler"] = yield_dataset.handler
    config["task"]["dataset"]["kwargs"]["segments"] = yield_dataset.segments
    print(yield_dataset.segments)
    # origin data -> MTS DataLoader
    yield_dataset = eval(config["task"]["dataset"]["class"])(**config["task"]["dataset"]["kwargs"])
    
    model = init_instance_by_config(config["task"]["model"])

    TOPK = config["strategy_config"]["topk"]
    # NDROP = 2
    BAD_THRESH = config["strategy_config"]["bad_thresh"]
    HT = config["strategy_config"]["hold_thresh"]

    strategy_config = {
        # "class": "TopkDropoutStrategy",
        # "module_path": "qlib.contrib.strategy.signal_strategy",
        "class": "TopkDropoutBadStrategy",
        "module_path": "lilab.qlib.strategy.signal_strategy",
        "kwargs": {
            "model": model,
            "dataset": pred_dataset,
            "topk": TOPK,
            "bad_thresh": BAD_THRESH,
            "hold_thresh": HT,
        },
    }

    EXP_NAME = config["task"]["model"]['kwargs']['model_type']
    URI = '/home/linq/finance/qniverse/mlrun'
    
    with R.start(experiment_name=EXP_NAME, uri=URI):
        rid = RID
        print("*******")
        print(rid)
        print("*******")
    
    port_analysis_config = {
        "executor": pred_benchmark.executor_config,
        "strategy": strategy_config,
        "backtest": pred_benchmark.backtest_config
    }
    

    # backtest and analysis
    with R.start(experiment_name=EXP_NAME, uri=URI):
        recorder = R.get_recorder(recorder_id=rid)
        model = recorder.load_object("trained_model")
        # prediction
        sr = SignalRecord(model, pred_dataset, recorder)
        sr.generate()
        # backtest & analysis
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
    
    tomorrow_return = model.predict(yield_dataset)
    tomorrow_return = tomorrow_return.sort_values(by='score', ascending=False)
    tomorrow_return.to_csv(f'../log/{EXP_NAME}_tomorrow_return_{TODAY}.csv')

    from qlib.contrib.report import analysis_model, analysis_position

    # load record
    with R.start(experiment_name=EXP_NAME, uri=URI):
        recorder = R.get_recorder(recorder_id=rid)
        pred_df = recorder.load_object("pred.pkl")
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

        recorder.save_objects(artifact_path='portfolio_analysis', **{'pred_label.pkl':pred_df})
        print(recorder)

    fig_list = analysis_position.report_graph(report_normal_df, show_notebook=False)
    for i, fig in enumerate(fig_list):
        fig: plotly.graph_objs.Figure = fig
        fig.write_image(f'../tmp/{EXP_NAME}_rg_{i}.jpg',engine='kaleido')

    
    fig_list = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
    for i, fig in enumerate(fig_list):
        fig: plotly.graph_objs.Figure = fig
        fig.write_image(f'../tmp/{EXP_NAME}_rag_{i}.jpg',engine='kaleido')
    


    label_df = pred_dataset.prepare("test", col_set="label")
    label_df.columns = ["label"]
    # pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

    fig_list = analysis_position.score_ic_graph(pred_df, show_notebook=False)

    for i, fig in enumerate(fig_list):
        fig: plotly.graph_objs.Figure = fig
        fig.write_image(f'../tmp/{EXP_NAME}_sig_{i}.jpg',engine='kaleido')


    fig_list = analysis_model.model_performance_graph(pred_df, show_notebook=False)
    for i, fig in enumerate(fig_list):
        fig: plotly.graph_objs.Figure = fig
        fig.write_image(f'../tmp/{EXP_NAME}_mpg_{i}.jpg',engine='kaleido')

    rank_df = pred_df.groupby('datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    rank_df.index = rank_df.index.droplevel(0)
    rank_df.to_csv(f'../log/{EXP_NAME}_backtest_return_{TODAY}.csv')

    top: pd.DataFrame = tomorrow_return.head(20)
    bottom: pd.DataFrame = tomorrow_return.tail(20)

    # top10.to_markdown(f'../tmp/{model_name}_top10.md')
    # bottom10.to_markdown(f'../tmp/{model_name}_bottom10.md')

    import dataframe_image as dfi

    dfi.export(top, f'../tmp/{model_name}_top10.png',table_conversion='matplotlib')
    dfi.export(bottom, f'../tmp/{model_name}_bottom10.png',table_conversion='matplotlib')

    infodf = pd.DataFrame({'label': ['Model update date', 'Prediction generation date', 'Top K', 'Sell thresh', 'Hold thresh'],
                       'value': ['2024-11-16', TODAY, TOPK, BAD_THRESH, HT]})

    dfi.export(infodf, f'../tmp/{model_name}_info.png',table_conversion='matplotlib')


if __name__ == "__main__":
    # set params from cmd
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--config_file", type=str, default="configs/config_timemixer.yaml", help="config file")
    # parser.add_argument("--today", type=str, default="2024-12-10")
    parser.add_argument("--btstart", type=str, default="2024-01-01")
    args = parser.parse_args()
    main(**vars(args))
