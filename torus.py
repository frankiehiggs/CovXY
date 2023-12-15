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
from itertools import combinations

@jit(nopython=True,parallel=False)
def sample_point( d, sample_size ):
    """
    Chooses a point uniformly in [0,1]^d.
    """
    return np.random.random(size=(sample_size,d))

def limit( beta, tau, d, k ):
    """
    Evaluates the limiting cdf from Theorem 2.1.
    """
    return np.exp( - tau*np.exp(-beta) / np.math.factorial(k-1) )

def corrected_limit(beta, tau, d, k, n):
    """
    Returns the "corrected limit" from Theorem 2.1.
    """
    correction = tau * np.exp(-beta) * (k-1)*(k-1) * np.log(np.log(n)) / ( np.math.factorial(k-1) * np.log(n) )
    return np.exp(-correction)*limit(beta,tau,d,k)

@jit(nopython=True)
def shift(torus_points,coords):
    shifted = torus_points.copy()
    for i in range(torus_points.shape[0]):
        for coord in coords:
            shifted[i,coord] = (shifted[i,coord] + 0.5) % 1.0
    return shifted

def lhs_quantity( R, n, k, d ):
    """
    Given R, calculates the random variable whose limit we consider in
    Theorem 2.1.
    """
    THETA_d = np.pi**(d/2) / np.math.gamma(d/2 + 1)
    f0 = 1
    return n*THETA_d*f0*R**d - np.log(n) - (k-1)*np.log(np.log(n))

@jit(nopython=True)
def generate_Rk(n,m,d,k):
    Xn = np.random.random(size=(n,d))
    Ym = np.random.random(size=(m,d))
    k_nearest_dists = np.empty(m)
    for j, y in enumerate(Ym):
        closest_k = d*np.ones(k)
        for x in Xn:
            too_far = False
            total_distance = 0
            for i in range(d):
                increment = np.abs(y[i] - x[i])
                if increment <= 0.5:
                    total_distance += increment*increment
                else:
                    total_distance += (1-increment)*(1-increment)
                if total_distance >= closest_k[-1]:
                    too_far = True
                    break # Seems to improve performance
            if too_far:
                continue
            else:
                new_pos = k-1
                while total_distance < closest_k[new_pos-1]:
                    closest_k[new_pos] = closest_k[new_pos-1]
                    new_pos -= 1
                    if new_pos == 0:
                        break
                closest_k[new_pos] = total_distance
        k_nearest_dists[j] = closest_k[-1]
    return np.sqrt(max(k_nearest_dists))

@jit
def generate_R_samples(n, m, d, k, number_of_samples=2):
    """
    Produces samples of the coverage threshold R_{n,m}.
    This function takes up the majority of the runtime.
    """
    samples = np.empty(number_of_samples)
    progress = tqdm(range(number_of_samples))
    if k == 1:
        for s in progress:
            Xn = sample_point(d, n)
            tree_0 = KDTree(Xn)
            Yn = sample_point(d, m)
            # Measure max_j min_i d(Y_j, X_i):
            distances = tree_0.query(Yn)[0]
            # This is quite a crude method for finding the closest points on a torus:
            # for each subset I of {1,...,d} we shift the points in the square by 0.5 * \sum_{i \in I} e_i,
            # then reduce the coordinates modulo 1, and measure distances in the square.
            # Especially in high dimensions it's slow, but it allows us to use numba_kdtree,
            # so ends up being faster than simply measuring pairwise distances on the torus.
            # For k > 1 we'll measure pairwise distances though. We could possibly do something
            # similar to this "translation" method to speed up the k > 1 case too.
            for r in range(1,d+1):
                for choice in combinations(range(d),r):
                    shifted_Xn   = shift(Xn,choice)
                    shifted_tree = KDTree(shifted_Xn)
                    shifted_Yn   = shift(Yn,choice)
                    distances = np.minimum(distances, shifted_tree.query(shifted_Yn)[0])
            samples[s] = np.max( distances )
    else:
        for s in progress:
            samples[s] = generate_Rk(n,m,d,k)
    return samples
