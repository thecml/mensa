import numpy as np
import pandas as pd
import torch
from scipy.special import erfinv
from pycop import simulation
from scipy import stats
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.allow_tf32 = False


def safe_log(x):
    return np.log(x+1e-20*(x<1e-20))


def linear_dgp_parametric_ph(copula_name='Frank', theta=10, n_samples=30000, n_features=10, rng=np.random.default_rng()):
    # Generate synthetic data (time-to-event and censoring indicator)
    # with linear parametric proportional hazards model (Weibull CoxPH)
    # This follows Ali's paper (Algorithm 2)
    def inverse_transform(value, risk, shape, scale):
        return (-safe_log(value) / np.exp(risk)) ** (1 / shape) * scale

    v_e=4; rho_e=14; v_c=3; rho_c=16

    # generate X from 10 dimensional uniform distribution from 0 to 1
    X = rng.uniform(0, 1, (n_samples, n_features))
    # generate censoring risk coefficient beta from 10 dimensional uniform distribution from 0 to 1
    beta_c = rng.uniform(0, 1, (n_features,))
    # generate event risk coefficient beta_e from 10 dimensional uniform distribution from 0 to 1
    beta_e = rng.uniform(0, 1, (n_features,))


    event_risk = np.matmul(X, beta_e).squeeze()
    censoring_risk = np.matmul(X, beta_c).squeeze()

    if copula_name=='Frank':
        u_e, u_c = simulation.simu_archimedean('frank', 2, n_samples, theta)
    elif copula_name=='Gumbel':
        u_e, u_c = simulation.simu_archimedean('gumbel', 2, n_samples, theta)
    elif copula_name=='Clayton':
        u_e, u_c = simulation.simu_archimedean('clayton', 2, n_samples, theta)
    elif copula_name=="Independent":
        u_e = rng.uniform(0, 1, n_samples)
        u_c = rng.uniform(0, 1, n_samples)
    else:
        raise ValueError('Copula not implemented')

    event_time = inverse_transform(u_e, event_risk, v_e, rho_e)
    censoring_time = inverse_transform(u_c, censoring_risk, v_c, rho_c)

    # create observed time
    observed_time = np.minimum(event_time, censoring_time)
    event_indicator = (event_time < censoring_time).astype(int)

    # to dataframe, include X, observed_time, event_indicator, event_time, censoring_time
    data = np.concatenate([X, observed_time[:, None], event_indicator[:, None], event_time[:, None],
                           censoring_time[:, None]], axis=1)
    columns = [f'X{i}' for i in range(n_features)] + ['observed_time', 'event_indicator', 'event_time',
                                                         'censoring_time']
    df = pd.DataFrame(data, columns=columns)
    return df, (beta_e, beta_c)

def relu(z):
    return np.maximum(0, z)

def inverse_transform_weibull(p, shape, scale):
    # transform CDF to x
    return scale * (-safe_log(p)) ** (1 / shape)

def inverse_transform_lognormal(p, shape, scale):
    return stats.lognorm(s=scale*0.25, scale=shape).ppf(p)

def inverse_transform_exp(p, shape, scale):
    return stats.expon(scale).ppf(p)




def dgp(copula_name='Frank', theta=10, n_samples=30000, n_features=10, rng=np.random.default_rng()):
    # Generate synthetic data (time-to-event and censoring indicator)
    # with parametric model (Weibull)
    hidden_dim = int(n_features/2)
    # generate X from 10 dimensional uniform distribution from 0 to 1
    X = rng.uniform(0, 1, (n_samples, n_features))
    # generate common coefficient betas
    beta = rng.uniform(0, 1, (n_features, hidden_dim))

    # generate censoring risk coefficient beta from 10 dimensional uniform distribution from 0 to 1
    beta_shape_c = rng.uniform(0, 1, (int(hidden_dim*0.8),))
    beta_scale_c = rng.uniform(0, 1, (int(hidden_dim*0.8),))
    # generate event risk coefficient beta_e from 10 dimensional uniform distribution from 0 to 1
    beta_shape_e = rng.uniform(0, 1, (int(hidden_dim*0.8),))
    beta_scale_e = rng.uniform(0, 1, (int(hidden_dim*0.8),))

    hidden_rep = np.matmul(X, beta).squeeze()
    # add non-linear transformation
    hidden_rep = relu(hidden_rep)
    shape_c = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_c).squeeze()
    scale_c = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_c).squeeze()
    shape_e = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_e).squeeze()
    scale_e = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_e).squeeze()

    # event_risk = np.matmul(hidden_rep, beta_e).squeeze()
    # censoring_risk = np.matmul(hidden_rep, beta_c).squeeze()

    if copula_name=='Frank':
        u_e, u_c = simulation.simu_archimedean('frank', 2, n_samples, theta)
    elif copula_name=='Gumbel':
        u_e, u_c = simulation.simu_archimedean('gumbel', 2, n_samples, theta)
    elif copula_name=='Clayton':
        u_e, u_c = simulation.simu_archimedean('clayton', 2, n_samples, theta)
    elif copula_name=="Independent":
        u_e = rng.uniform(0, 1, n_samples)
        u_c = rng.uniform(0, 1, n_samples)
    else:
        raise ValueError('Copula not implemented')

    event_time = inverse_transform_weibull(u_e, shape_e, scale_e)
    censoring_time = inverse_transform_weibull(u_c, shape_c, scale_c)

    # create observed time
    observed_time = np.minimum(event_time, censoring_time)
    event_indicator = (event_time < censoring_time).astype(int)

    # to dataframe, include X, observed_time, event_indicator, event_time, censoring_time
    data = np.concatenate([X, observed_time[:, None], event_indicator[:, None], event_time[:, None],
                           censoring_time[:, None]], axis=1)
    columns = [f'X{i}' for i in range(n_features)] + ['observed_time', 'event_indicator', 'event_time',
                                                         'censoring_time']
    df = pd.DataFrame(data, columns=columns)
    return df, (beta, beta_shape_e, beta_scale_e)


def dgp_cr(copula_parameters: list, dist='Weibull', n_samples=30000, n_features=10, rng=np.random.default_rng()):
    # Generate synthetic data (two competing risks and censoring)
    if copula_parameters is None:
        corrMatrix = np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]])

        copula_parameters = [
            {"type": "clayton", "weight": 1 / 3, "theta": 2},
            {"type": "student", "weight": 1 / 3, "corrMatrix": corrMatrix, "nu": 2},
            {"type": "gumbel", "weight": 1 / 3, "theta": 3}
        ]

    # generate X from 10 dimensional uniform distribution from 0 to 1
    X = rng.uniform(0, 1, (n_samples, n_features))
    hidden_dim = int(n_features/2)
    # generate X from 10 dimensional uniform distribution from 0 to 1
    X = rng.uniform(0, 1, (n_samples, n_features))
    # generate common coefficient betas
    beta = rng.uniform(0, 1, (n_features, hidden_dim))

    # generate censoring risk coefficient beta
    beta_shape_c = rng.uniform(0, 1, (int(hidden_dim*0.8),))
    beta_scale_c = rng.uniform(0, 1, (int(hidden_dim*0.8),))
    # generate event risk coefficient beta_e1
    beta_shape_e1 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
    beta_scale_e1 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
    # generate event risk coefficient beta_e2
    beta_shape_e2 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
    beta_scale_e2 = rng.uniform(0, 1, (int(hidden_dim*0.8),))

    hidden_rep = np.matmul(X, beta).squeeze()
    # add non-linear transformation
    hidden_rep = relu(hidden_rep)

    shape_c = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_c).squeeze()
    scale_c = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_c).squeeze()
    shape_e1 = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_e1).squeeze()
    scale_e1 = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_e1).squeeze()
    shape_e2 = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_e2).squeeze()
    scale_e2 = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_e2).squeeze()

    u_e1, u_e2, u_c = simulation.simu_mixture(3, n_samples, copula_parameters)

    if dist == "Weibull":
        e1_time = inverse_transform_weibull(u_e1, shape_e1, scale_e1)
        e2_time = inverse_transform_weibull(u_e2, shape_e2, scale_e2)
        c_time = inverse_transform_weibull(u_c, shape_c, scale_c)
    elif dist == "Exp":
        e1_time = inverse_transform_exp(u_e1, shape_e1, scale_e1)
        e2_time = inverse_transform_exp(u_e2, shape_e2, scale_e2)
        c_time = inverse_transform_exp(u_c, shape_c, scale_c)
    elif dist == "Lognormal":
        e1_time = inverse_transform_lognormal(u_e1, shape_e1, scale_e1)
        e2_time = inverse_transform_lognormal(u_e2, shape_e2, scale_e2)
        c_time = inverse_transform_lognormal(u_c, shape_c, scale_c)
    else:
        raise ValueError('Dist not implemented')

    # create observed time
    # concatenate event times and censoring times
    all_times = np.stack([e1_time, e2_time, c_time], axis=1)
    observed_time = np.min(all_times, axis=1)
    event_indicator = np.argmin(all_times, axis=1)

    # to dataframe, include X, observed_time, event_indicator, e1_time, e2_time, c_time
    data = np.concatenate([X, observed_time[:, None], event_indicator[:, None], e1_time[:, None], e2_time[:, None],
                           c_time[:, None]], axis=1)
    columns = [f'X{i}' for i in range(n_features)] + ['observed_time', 'event_indicator', 'e1_time', 'e2_time', 'c_time']

    df = pd.DataFrame(data, columns=columns)
    return df, (beta, beta_shape_e1, beta_scale_e1, beta_shape_e2, beta_scale_e2)


if __name__ == '__main__':
    # df, params = linear_dgp_parametric_ph(copula_name='Frank', n_samples=30000, n_features=10, theta=10)
    # df, params = dgp(copula_name='Frank', n_samples=30000, n_features=10, theta=10)
    df, params = dgp_cr(copula_parameters=None, dist="Weibull",
                        n_samples=30000, n_features=10)

    print(df.head())
    import matplotlib.pyplot as plt
    e1_times = df['e1_time']
    e2_times = df['e2_time']
    
    plt.hist(e1_times, bins=20, alpha=0.5, label='Data 1', edgecolor='black')
    plt.hist(e2_times, bins=20, alpha=0.5, label='Data 1', edgecolor='black')
    plt.show()
    