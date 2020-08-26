import scipy as sp
import numpy as np
import torch

def Gmatrix(B):
    return np.eye(B) - np.ones([B, B]) / B

def paramdot(d1, d2):
    return sum(
        torch.dot(d1[k].reshape(-1), d2[k].reshape(-1))
        for k in d1)
def clone_grads(net):
    d = {}
    for name, p in net.named_parameters():
        if p.grad is not None:
            d[name] = p.grad.clone().detach()
    return d
def flatten(grads):
    grads = dict(grads)
    g = []
    for k, v in grads.items():
        g.append(v.reshape(-1))
    return torch.cat(g)
def getCov(x, normalize=True):
    C = x @ x.T
    if normalize:
        return C / x.shape[-1]
    else:
        return C

def getCor(cov):
    d = np.diag(cov)**-0.5
    return d[:, None] * cov * d

# V-Transforms

def VStep(cov):
    '''
    Computes E[step(z) step(z)^T | z ~ N(0, `cov`)]
    where step is the function takes positive numbers to 1 and
    all else to 0, and 
    z is a multivariate Gaussian with mean 0 and covariance `cov`
    
    Inputs:
        `cov`: An array where the last 2 dimensions contain covariance matrix of z (and the first dimensions are "batch" dimensions)
    Output:
        a numpy array of the same shape as `cov` that equals the
        expectation above in the last 2 dimensions.
    '''
    ll = list(range(cov.shape[-1]))
    d = np.sqrt(cov[..., ll, ll])
    c = d[..., None]**(-1) * cov * d[..., None, :]**(-1)
    return (0.5 / np.pi) * (np.pi - np.arccos(np.clip(c, -1, 1)))

def J1(c, eps=1e-10):
    c[c > 1-eps] = 1-eps
    c[c < -1+eps] = -1+eps
    return (np.sqrt(1-c**2) + (np.pi - np.arccos(c)) * c) / np.pi

def VReLU(cov, eps=1e-5):
    ll = list(range(cov.shape[-1]))
    d = np.sqrt(cov[..., ll, ll])
    c = d[..., None]**(-1) * cov * d[..., None, :]**(-1)
    return np.nan_to_num(0.5 * d[..., None] * J1(c, eps=eps) * d[..., None, :])
    
def VErf(cov):
    '''
    Computes E[erf(z) erf(z)^T | z ~ N(0, `cov`)]
    where z is a multivariate Gaussian with mean 0 and covariance `cov`

    Inputs:
        `cov`: An array where the last 2 dimensions contain covariance matrix of z (and the first dimensions are "batch" dimensions)
    Output:
        a numpy array of the same shape as `cov` that equals the
        expectation above in the last 2 dimensions.
    '''
    ll = list(range(cov.shape[-1]))
    d = np.sqrt(cov[..., ll, ll] + 0.5)

    c = d[..., None]**(-1) * cov * d[..., None, :]**(-1)
    return 2./np.pi * np.arcsin(np.clip(c, -1, 1))


def VDerErf(cov):
    '''
    Computes E[erf'(z) erf'(z)^T | z ~ N(0, `cov`)]
    where erf' is the derivative of erf and
    z is a multivariate Gaussian with mean 0 and covariance `cov`

    Inputs:
        `cov`: An array where the last 2 dimensions contain covariance matrix of z (and the first dimensions are "batch" dimensions)
    Output:
        a numpy array of the same shape as `cov` that equals the
        expectation above in the last 2 dimensions.
    '''
    ll = list(range(cov.shape[-1]))
    d = np.sqrt(cov[..., ll, ll])
    dd = 1 + 2 * d
    return 4/np.pi * (dd[..., None] * dd[..., None, :] - 4 * cov**2)**(-1./2)

# torch versions
def thJ1(c, eps=1e-10):
    c = torch.clamp(c, -1+eps, 1-eps)
    return ((1-c**2)**0.5 + (np.pi - torch.acos(c)) * c) / np.pi

def thVReLU(cov, eps=1e-6):
    ll = list(range(cov.shape[-1]))
    d = cov[..., ll, ll]**0.5
    c = d[..., None]**(-1) * cov * d[..., None, :]**(-1)
    out = d[..., None] * thJ1(c, eps=eps) * d[..., None, :]
    out[..., ll, ll] = cov[..., ll, ll]
    return 0.5 * out

# triplet versions
def VErf3(cov, v, v2=None, eps=1e-7):
    '''
    Computes E[erf(z1)erf(z2)]
    where (z1, z2) is sampled from a 2-dimensional Gaussian
    with mean 0 and covariance
    |v    cov|
    |cov  v  |
    or
    |v    cov|
    |cov  v2 |
    
    Inputs:
        `cov`: covariance of input matrix
        `v`: common diagonal variance of input matrix, if `v2` is None;
            otherwise, the first diagonal variance
        `v2`: the second diagonal variance
    The inputs can be tensors, in which case they need to have the same shape
    '''
    if v2 is not None:
        return 2/np.pi * np.arcsin(cov/np.sqrt((v+0.5) * (v2+0.5)) + eps)
    else:
        return 2/np.pi * np.arcsin(cov/(v+0.5) + eps)
def VDerErf3(cov, v, v2=None, eps=1e-7):
    '''
    Computes E[erf'(z1)erf'(z2)]
    where (z1, z2) is sampled from a 2-dimensional Gaussian
    with mean 0 and covariance
    |v    cov|
    |cov  v  |
    or
    |v    cov|
    |cov  v2 |
    
    Inputs:
        `cov`: covariance of input matrix
        `v`: common diagonal variance of input matrix, if `v2` is None;
            otherwise, the first diagonal variance
        `v2`: the second diagonal variance
    The inputs can be tensors, in which case they need to have the same shape
    '''
    if v2 is not None:
        return 4/np.pi * ((1+2*v)*(1+2*v2) - 4 * cov**2 + eps)**-0.5
    else:
        return 4/np.pi * ((1+2*v)**2 - 4 * cov**2 + eps)**-0.5