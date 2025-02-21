# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from typing import List, Text, Tuple, Union
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
from qlib.workflow import R

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

from gpflow.kernels import RBF, White, Matern32
from gpflow.likelihoods import Gaussian
from gpflow.models.svgp import SVGP
from gpflow.training import AdamOptimizer, ScipyOptimizer
from scipy.stats import mode
from scipy.cluster.vq import kmeans2
import gpflow
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow import autoflow, params_as_tensors, ParamList
import pandas as pd
import itertools
pd.options.display.max_rows = 999
from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.io import loadmat
# from gpflow_monitor import *
print('tf_ver:', tf.__version__, 'gpflow_ver:', gpflow.__version__)
from tensorflow.python.client import device_lib
print('avail devices:\n'+'\n'.join([x.name for x in device_lib.list_local_devices()]))
# from jack_utils.common import time_it
import sys, pickle
import gpflow.training.monitor as mon
from numpy import exp, log

from scipy.stats import spearmanr, kendalltau
from ggd import GGD

from doubly_stochastic_dgp.dgp import DGP

from gpflow.training import AdamOptimizer, ScipyOptimizer, NatGradOptimizer
from gpflow.actions import Action, Loop
from qlib.log import get_module_logger

class G4PModel(Model):

    def __init__(self, M, L, feat_dim, 
        minibatch_size,
        num_samples,
        iterations, 
        llh,
        kernel,
        ARD,
        eval_step,
        cov_type,
        **kwargs):
        self.params = {"verbosity": -1}
        self.params.update(kwargs)
        
        self.M = M
        self.L = L
        self.feat_dim = feat_dim
        self.minibatch_size = minibatch_size
        self.num_samples = num_samples
        self.iterations = iterations
        if llh=='GGD':
            self.llhf = GGD(p=2.0)
        else:
            self.llhf = Gaussian()
        self.kernel = kernel
        self.ARD = ARD
        self.eval_step = eval_step
        self.cov_type = cov_type

    def _prepare_data(self, dataset: DatasetH, key = 'train'):
        if key == "train":
            df = dataset.prepare(key, data_key=DataHandlerLP.DK_L)
            if df.empty:
                raise ValueError("Empty data from dataset, please check your dataset config.")
            x, y = df.iloc[:,:-1], df.iloc[:,-1]
            self.X_raw, self.Y_raw = x.values, y.values.reshape((-1,1))

            self.X_mu, self.X_std = x.mean().values, x.std().values
            self.Y_mu, self.Y_std = y.mean(), y.std()
            self.X_norm = ((x - self.X_mu) / self.X_std).values
            self.Y_norm = ((y - self.Y_mu) / self.Y_std).values.reshape((-1,1))            
            return self.X_norm, self.Y_norm
        elif key == "test":
            df = dataset.prepare(key, data_key=DataHandlerLP.DK_I)
            if df.empty:
                raise ValueError("Empty data from dataset, please check your dataset config.")
            x, y = df.iloc[:,:-1], df.iloc[:,-1]
            X_norm = ((x - self.X_mu) / self.X_std).values
            Y_norm = ((y - self.Y_mu) / self.Y_std).values.reshape((-1,1))            
            return df.index, X_norm, Y_norm       
        

    def batch_assess(self, model, assess_model, X, Y = None):
        n_batches = max(int(X.shape[0]/self.minibatch_size), 1)
        lik, sq_diff, mlst, vlst = [], [], [], []
        if Y is not None:
            for X_batch, Y_batch in zip(np.array_split(X, n_batches), np.array_split(Y, n_batches)):
                l, sq, m, v = assess_model(model, X_batch, Y_batch)
                lik.append(l)
                sq_diff.append(sq)
                mlst.append(m)
                vlst.append(v)
            lik = np.concatenate(lik, 0)
            sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
            lik = np.average(lik)
            sq_diff = np.average(sq_diff)**0.5
        else:
            for X_batch in np.array_split(X, n_batches):
                l, sq, m, v = assess_model(model, X_batch)
                mlst.append(m)
                vlst.append(v)
            lik = 0
            sq_diff = 0
        m = np.concatenate(mlst, 0)
        v = np.concatenate(vlst, 0)
        return lik, sq_diff, m*self.Y_std+self.Y_mu, v*self.Y_std**2

    # def assess_single_layer(self, model, X_batch, Y_batch):
    #     Y_std = self.Y_std
    #     m, v = model.predict_y(X_batch)
    #     lik = np.sum(norm.logpdf(Y_batch*Y_std, loc=m*Y_std, scale=Y_std*v**0.5),  1)
    #     sq_diff = Y_std**2*((m - Y_batch)**2)
    #     return lik, sq_diff 

    
    def assess_sampled(self, model, X_batch, Y_batch = None):
        Y_std = self.Y_std
        S = self.num_samples
        m, v = model.predict_y(X_batch, S)
        
        if Y_batch is None:
            lik, sq_diff = 0, 0
        else:
            S_lik = np.sum(norm.logpdf(Y_batch*Y_std, loc=m*Y_std, scale=Y_std*v**0.5), 2)
            lik = logsumexp(S_lik, 0, b=1/float(S))        
            mean = np.average(m, 0)
            sq_diff = Y_std**2*((mean - Y_batch)**2)
        m = np.mean(m, 0)
        v = np.mean(v, 0)
        return lik, sq_diff, m, v 
    
    def build(self, dataset):
        logger = get_module_logger('G4PQ')
        minibatch_size = self.minibatch_size
        iterations = self.iterations
        X, Y = self._prepare_data(dataset)
        logger.info(f'Loading training data Done. X.shape: {X.shape}')
        # Z = kmeans2(X, self.M, minit='points')[0] 
        Z = np.random.randn(self.M, X.shape[1])
        D = self.feat_dim
        L = self.L
        kernels = []
        for l in range(L):
            kernels.append(eval(f'{self.kernel}(D, ARD=self.ARD)'))

        # for kernel in kernels[:-1]:
        #     kernel += White(D, variance=2e-6)

        mb = minibatch_size if X.shape[0] > minibatch_size else X.shape[0]
        with gpflow.defer_build():
            self.DGPModel = m = DGP(X, Y, Z, kernels, self.llhf, num_samples=self.num_samples, minibatch_size=mb)
        m.compile()
        return m

    def fit(
        self,
        dataset: DatasetH,
        **kwargs,
    ):
        logger = get_module_logger('G4PQ')
        minibatch_size = self.minibatch_size
        iterations = self.iterations
        X, Y = self._prepare_data(dataset)

        m = self.build(dataset)
        for layer in m.layers[:-1]:
            layer.q_sqrt = layer.q_sqrt.value * 1e-5

        s = 'Opt step: {}, lik: {:.4f}, rmse: {:.4f}'
        # for iterations in [iterations_few, iterations_many]:
        # print('after {} iterations'.format(iterations))
        # ng_vars = [[m.layers[-1].q_mu, m.layers[-1].q_sqrt]]
        # for v in ng_vars[0]:
        #     v.set_trainable(False)    
        # ng_action = NatGradOptimizer(gamma=0.1).make_optimize_action(m, var_list=ng_vars)
        
        logger.info('Training Start')
        optimiser = gpflow.train.AdamOptimizer(0.01)
        def my_callback(step, **args):
            logger.info(f'Opt step: {step}')     
            if step > 0 and step % self.eval_step == 0:
                # s = 'lik: {:.4f}, rmse: {:.4f}'          
                # logger.info('Eval Start') 
                lik, rmse, _, _ = self.batch_assess(m, self.assess_sampled, X, Y)
                logger.info(s.format(step, lik, rmse))
        optimiser.minimize(m, maxiter=iterations, step_callback=my_callback)

        lik, rmse, m, v = self.batch_assess(m, self.assess_sampled, X, Y)
        
        logger.info('Training Done')
        logger.info('lik: {:.4f}, rmse: {:.4f}'.format(lik, rmse))
        self.train_lik, self.train_rmse = lik, rmse


    def predict(self, dataset: DatasetH, segment = "test", online = False):
        logger = get_module_logger('G4PQ')        
        cov_type = self.cov_type
        idx, X, Y = self._prepare_data(dataset, key='test')
        if online:
            Y = None
        logger.info(f'Loading test data Done. X.shape: {X.shape}')
        lik, rmse, m, v = self.batch_assess(self.DGPModel, self.assess_sampled, X, Y)
        logger.info('Predict Done')
        if not online:
            logger.info('lik: {:.4f}, rmse: {:.4f}'.format(lik, rmse))
        if cov_type is None:
            m_res = pd.Series(m.reshape(-1), index=idx)
            return m_res
        elif cov_type == 'var':
            m_res = pd.DataFrame({'mean': m.reshape(-1), 'var': v.reshape(-1)}, index=idx)
            return m_res
