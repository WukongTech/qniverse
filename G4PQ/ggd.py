import numpy as np
import tensorflow as tf

from gpflow.likelihoods import MonteCarloLikelihood
from gpflow import logdensities
from gpflow import priors
from gpflow import settings
from gpflow import transforms
from gpflow.decors import params_as_tensors
from gpflow.decors import params_as_tensors_for
from gpflow.params import ParamList
from gpflow.params import Parameter
from gpflow.params import Parameterized
from gpflow.quadrature import hermgauss
from gpflow.quadrature import ndiagquad, ndiag_mc

from tensorflow.math import exp, abs, log, sin, lgamma, pow
from numpy import pi

class GGD(MonteCarloLikelihood):
    def __init__(self, p=1.0, alpha=1.0, K=100, **kwargs):
        super().__init__(**kwargs)
        self.alpha = Parameter(
            alpha, transform=transforms.positive, dtype=settings.float_type)
        self.p = Parameter(
            p, transform=transforms.positive, dtype=settings.float_type)
        self.K = K
#         self.p = tf.cast(p, settings.float_type)

    def logp(self, F, Y):
        def log_ggd(x, p, mu, alpha):
            cp = tf.cast(log(p) - ((p+1)/p)*tf.cast(log(2.0), settings.float_type) - lgamma(1/p), settings.float_type)
            res = tf.cast(cp - log(alpha) - pow(abs(x-mu), p)/(2*pow(alpha,p)), settings.float_type)
            return res
        res = log_ggd(Y, self.p, F, self.alpha)
        return res
    
    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        alpha = self.alpha
        p = self.p
        tmp = 4 * pow(alpha,2) * exp(lgamma(3/p)-lgamma(1/p))
        return tf.fill(tf.shape(F), tf.squeeze(tmp))
    
    def predict_mean_and_var(self, Fmu, Fvar, epsilon=None):
        integrand2 = lambda *X: self.conditional_variance(*X) + tf.square(self.conditional_mean(*X))
        E_y, E_y2 = self._mc_quadrature([self.conditional_mean, integrand2],
                                        Fmu, Fvar, epsilon=epsilon)
        V_y = E_y2 - tf.square(E_y)
        return tf.identity(Fmu), V_y  # N x D