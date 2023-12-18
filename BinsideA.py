"""
Samples R_{m,n,k} in the setting of Theorem 2.1,
i.e. covering B with closure(B) \subseteq interior(A).
In particular we cover B(0, 0.9) using points placed in B(0, 1).
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
from ball import theta, c_dk, sigma_A, sample_point

def limit( beta, tau, k ):
    """
    Evaluates the limiting cdf from Theorem 2.1.
    This is not the limiting cdf of R_{n,m,k} itself,
    but of the derived quantity n theta(d) f_0 R^d - log...
    """
    return np.exp( - tau * np.exp(-beta) / np.math.factorial(k-1) )

def corrected_limit(beta, tau, k, n):
    """
    Returns the "corrected limit" from Theorem 2.1.
    """
    correction = tau * np.exp(-beta) * (k-1)**2 * np.log(np.log(n)) / ( np.math.factorial(k-1) * np.log(n) )
    return np.exp(-correction)*limit(beta,tau,k)

def lhs_quantity( R, n, k, d ):
    """
    Given R, calculates the random variable
    whose limit we consider in Theorem 2.1.
    """
    THETA_d = theta(d)
    f0 = 1/THETA_d
    return n*THETA_d*f0*(R**d) - np.log(n) - (k-1)*np.log(np.log(n))

@jit
def generate_R_samples(n, m, d, k, number_of_samples=2, shrinkage_factor=0.9):
    """
    Produces samples of the two-sample coverage threshold.
    This function takes up the majority of the runtime.
    It returns lhs_quantity(R_{n,m,k}),
    not R_{n,m,k} itself.
    """
    samples = np.empty(number_of_samples)
    progress = tqdm(range(number_of_samples))
    if k == 1:
        for s in progress:
            progress.set_description("Step 1/4: sampling n points")
            Xn = sample_point(d, n)
            progress.set_description("Step 2/4: building k-d tree")
            tree = KDTree(Xn)
            progress.set_description("Step 3/4: sampling m points")
            Yn = shrinkage_factor*sample_point(d, m)
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
            Yn = shrinkage_factor*sample_point(d, m)
            progress.set_description("Step 4/4: measure distances")
            distances = tree.query(Yn,k=k)[0][:,k-1]
            samples[s] = np.max( distances )
    return lhs_quantity(samples,n,k,d)
