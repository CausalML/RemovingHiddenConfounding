from sklearn.cluster import SpectralClustering as sc
import numpy.random as rand
from numpy.linalg import norm as dist
import numpy as np
from functools import partial
from scipy.special import expit


def generate_data_with_linear_eta(obs_args, kappa, ft_args, m, sigma = 0.5):
    """generate data for the setting where corr(X,U) = kappa(2T-1)

    :obs_args: TODO
    :kappa: TODO
    :ft_args: TODO
    :m: TODO
    :returns: TODO

    """
    mu_0, mu_1, nu_0, nu_1 = obs_args
    assert kappa*(mu_1 - mu_0) == nu_1 + nu_0, "E[U|X] condition is not true"
    ft = lambda t,x : alpha[t] + beta[t]*x + gamma[t]*x*x

    print('generating Obs data')
    mean = [[mu_0, nu_0],[mu_1, nu_1]]
    cov = (2*T-1)*kappa
    X, U = np.zeros(cov.shape), np.zeros(cov.shape[0])
    for i in range(T.shape[0]):
        X[i], U[i] = np.random.multivariate_normal(mean[T[i]], [[1, cov[i]],[cov[i],1]]).T
    Y = np.vectorize(ft)(T,X) + U + sigma* np.random.normal(size=U.shape)
    print('generating exp data')
    X_E = np.random.uniform(-1,1, size=(m,))
    U_E = np.random.normal(0,1, size=(m,))
    T_E = rand.randint(0,2,(m,))
    Y_E = np.vectorize(ft)(T_E, X_E) + U_E + sigma* np.random.normal(size=U_E.shape)

    return X, Y, T, X_E, Y_E, T_E
