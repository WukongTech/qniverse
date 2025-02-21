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

from lilab.qlib.utils.tools import get_return_dataframe

class UPAdapter:
    def __init__(self, market, st_time, ed_time, algo):
        handler_kwargs = {
            "start_time": st_time,
            "end_time": ed_time,
            "instruments": market,
            'infer_processors': [
                {
                'class': 'Fillna',
                }
            ],
            'data_loader': {
                "class": "QlibDataLoader",
                "module_path": "qlib.data.dataset.loader",
                "kwargs": {
                    'config': (
                        ['$close'],
                        ['rt']
                    )
                },
            }

        }
        handler_conf = {
            "class": "DataHandlerLP",
            "module_path": "qlib.data.dataset.handler",
            "kwargs": handler_kwargs,
        }
        hd = init_instance_by_config(handler_conf)


        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": hd,
                "segments": {
                    "train": (st_time, ed_time),
                    "valid": (st_time, ed_time),
                    "test": (st_time, ed_time),
                },
            },
        }

        dataset = init_instance_by_config(dataset_config)
        train_df = dataset.prepare('train')
        self.data = get_return_dataframe(train_df)
        self.data = self.data.drop(columns=self.data.columns[self.data.isin([np.nan, np.inf, -np.inf]).any()])
        # self.data = self.data.droplevel(0, 1)

        self.algo: up.Algo = algo

    def predict(self):
        result = self.algo.run(self.data)
        print(result.summary())

        self.result = result

    def get_weight(self):
        return self.result.B
    
    def get_qlib_weight(self):
        df = self.get_weight()
        df_reshaped = df.stack().reset_index()
        df_reshaped.columns = ['trade_dt', 'ticker', 'weight']
        df_reshaped = df_reshaped.set_index(['trade_dt', 'ticker'])
        df_reshaped = df_reshaped[df_reshaped['weight'] > 0]
        return df_reshaped