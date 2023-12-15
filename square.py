"""
Computes R_{n,m,k} in the 2-dimensional square [0,1]^2.
"""
import numpy as np
import matplotlib.pyplot as plt
#from scipy.spatial import KDTree
from numba_kdtree import KDTree
from scipy.stats import norm
from scipy.stats import binom
from scipy.stats import poisson
from scipy.integrate import quad
from tqdm import tqdm
import sys
from numba import jit
from ball import c_dk

@jit(nopython=True,parallel=False)
def sample_point( sample_size ):
    """
    Chooses a point uniformly in the square [0,1]^2.
    """
    return np.random.random(size=(sample_size,2))

def limit( beta, tau, k ):
    """
    This is not the limiting cdf of R_{n,m} itself,
    but of the derived quantity n pi R^2 - log...
    """
    SIGMA_A = 4
    C_dk = c_dk(2,k)
    if k == 1:
        # Theorem 2.2
        return np.exp( - tau*np.exp(-beta) )
    elif k == 2:
        # Theorem 2.3, equation (2.7)
        return np.exp( -tau*(np.exp(-beta) + 0.25*np.sqrt(np.pi)*SIGMA_A*np.exp(-0.5*beta) ))
    else:
        # Theorem 2.3, equation (2.9)
        return np.exp( - C_dk * tau * SIGMA_A * np.exp(-0.5*beta) )

def corrected_limit(beta, tau, k, n):
    """
    Returns the "corrected limit" from Theorems 2.2 and 2.3.
    """
    C_dk = c_dk(2,k)
    SIGMA_A = 4
    if k == 1:
        # Theorem 2.2
        correction = tau * np.sqrt(np.pi) * SIGMA_A * np.exp(-0.5*beta) / (2 * np.sqrt(np.log(n)))
    elif k == 2:
        # Theorem 2.3, equation (2.7)
        correction = tau * np.sqrt(np.pi) * SIGMA_A * np.exp(-0.5*beta) * np.log(np.log(n)) / (8 * np.log(n))
    else:
        # Theorem 2.3, equation (2.9)
        correction = C_dk*tau*SIGMA_A*np.exp(-0.5*beta)*(k-2 + 1/d)**2 * np.log(np.log(n)) / ( (1 - 1/d)*np.log(n))
    return np.exp(-correction)*limit(beta,tau, k)

def lhs_quantity( R, n, k ):
    """
    Given R, calculates the random variable whose limit we consider in
    Theorems 2.2 and 2.3.
    """
    f0 = 1
    if k == 1:
        return n*np.pi*f0*np.square(R) - np.log(n)
    else:
        return n*np.pi*f0*np.square(R) - np.log(n) - (2*k - 3)*np.log(np.log(n))

@jit
def generate_R_samples(n, m, k, number_of_samples=2):
    """
    Produces samples of the coverage threshold R_{n,m}.
    This function takes up the majority of the runtime.
    """
    samples = np.empty(number_of_samples)
    progress = tqdm(range(number_of_samples))
    if k == 1:
        for s in progress:
            progress.set_description("Step 1/4: sampling n points")
            Xn = sample_point(n)
            progress.set_description("Step 2/4: building k-d tree")
            tree = KDTree(Xn)
            progress.set_description("Step 3/4: sampling m points")
            Yn = sample_point(m)
            # Measure max_j min_i d(Y_j, X_i):
            progress.set_description("Step 4/4: measure distances")
            distances = tree.query(Yn)[0]
            samples[s] = np.max( distances )
    else:
        for s in progress:
            progress.set_description("Step 1/4: sampling n points")
            Xn = sample_point(d, n)
            progress.set_description("Step 2/4: building k-d tree")
            tree = KDTree(Xn)
            progress.set_description("Step 3/4: sampling m points")
            Yn = sample_point(d, m)
            progress.set_description("Step 4/4: measure distances")
            distances = tree.query(Yn,k=k)[0][:,k-1]
            samples[s] = np.max( distances )
    return samples
