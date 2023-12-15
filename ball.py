"""
Code used for the simulations in the paper
"Covering one point process with another"
by Frankie Higgs, Mathew D Penrose and Xiaochuan Yang.

Samples R_{n, tau n, k} for point processes placed inside a unit ball B(o,1).
We have separate files for points placed inside a square,
one for the case closure(B) \subset interior(A) described in the paper,
and another for points in a torus.
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

# Derived quantities
def theta(d):
    """
    Returns the volume of the d-dimensional unit ball.
    """
    return np.pi**(d/2) / np.math.gamma(d/2 + 1)

def c_dk(d,k):
    if k==1:
        if d==2:
            return 1
        else:
            return (1/theta(d-1)) * (theta(d)/(2 - 2/d))**(1 - 1/d)
    else:
        return (theta(d)**(1 - 1/d) * (1 - 1/d)**(k-2+ 1/d))/(np.math.factorial(k-1)*2**(1-1/d) * theta(d-1))

def sigma_A(d):
    return d * (theta(d)**(1/d))

@jit(nopython=True,parallel=False)
def sample_point( d, sample_size ):
    """
    Chooses a point uniformly in the unit ball.
    """
    if d==2:
        theta = 2*np.pi*np.random.random(size=(sample_size,1))
        radius_sqrt = np.sqrt(np.random.random(size=(sample_size,1)))
        x = radius_sqrt * np.cos(theta)
        y = radius_sqrt * np.sin(theta)
        samples = np.concatenate((x,y),axis=1)
    elif d<=4:
        # If d isn't too large then the fastest way to sample a point in the unit ball
        # is to repeatedly sample a point in the box [-1,1]^d
        # until we get a point inside the unit ball.
        samples = 2*np.random.random(size=(sample_size,d)) - 1
        for i in range(sample_size):
            while np.dot(samples[i,:],samples[i,:]) > 1:
                samples[i,:] = 2*np.random.random(size=d) - 1
    else:
        # uniform sample in the (solid) d-dimensional unit ball
        # Using "polar coordinates":
        # We choose a point uniformly on the unit sphere,
        # then multiply it by U^{1/d},
        # where U ~ U[0,1] indep of the point on the sphere.
        samples = np.empty(shape=(sample_size,d))
        for i in range(sample_size):
            row = np.empty(d)
            for j in range(d):
                row[j] = np.random.normal()
            row /= np.linalg.norm(row)
            row *= np.random.random()**(1/d)
            samples[i,:] = row
    return samples

def limit( beta, tau, d, k, subtract_median = False ):
    """
    Evaluates the limiting cdfs from Theorem 2.1, 2.2 and 2.3.

    This is not the limiting cdf of R_{n,m} itself,
    but of the derived quantity n theta(d) f_0 R^d - log...
    """
    SIGMA_A = sigma_A(d)
    C_dk = c_dk(d,k)
    if d==2 and k==1:
        # Theorem 2.2
        if subtract_median:
            beta = beta + np.log(tau) - np.log(np.log(2))
        return np.exp(-tau*np.exp(-beta))
    elif d==2 and k==2:
        # Theorem 2.3, equation (2.7)
        if subtract_median:
            median = -2*np.log( -0.125*np.sqrt(np.pi)*SIGMA_A + np.sqrt(np.pi*SIGMA_A*SIGMA_A/64 + np.log(2)/tau) )
            beta = beta + median
        return np.exp(-tau*( np.exp(-beta) + 0.25*np.sqrt(np.pi)*SIGMA_A*np.exp(-0.5*beta) ) )
    else:
        # Theorem 2.3, equation (2.9)
        if subtract_median:
            median = 2*( np.log( tau * C_dk * SIGMA_A ) - np.log(np.log(2)) )
            beta = beta + median
        return np.exp(-tau*C_dk*SIGMA_A*np.exp(-0.5*beta))

def corrected_limit(beta, tau, d, k, n,subtract_median=False):
    """
    Returns the "corrected limit" from Theorems 2.2 and 2.3
    """
    SIGMA_A = sigma_A(d)
    C_dk = c_dk(d,k)
    if d==2 and k==1:
        # Theorem 2.2
        if subtract_median:
            median = -2*np.log( np.sqrt( np.log(2)/tau + np.pi*SIGMA_A*SIGMA_A/(16*np.log(n)) ) - np.sqrt(np.pi)*SIGMA_A/(4*np.sqrt(np.log(n))) )
            # The above is the median of the corrected cdf
            beta = beta + median
        correction = tau * np.sqrt(np.pi) * SIGMA_A * np.exp(-0.5*beta) * 0.5 / np.sqrt(np.log(n))
    elif d==2 and k==2:
        # Theorem 2.3, equation (2.7)
        if subtract_median:
            r = 0.125*np.sqrt(np.pi)*SIGMA_A*(1 + 0.5*np.log(np.log(n))/np.log(n))
            median = -2*np.log( np.sqrt(r*r + np.log(2)/tau ) - r )
            # The above is the median of the corrected cdf
            beta = beta + median
        correction = tau * np.sqrt(np.pi) * SIGMA_A * np.exp(-0.5*beta) * np.log(np.log(n)) / (8 * np.log(n))
    else:
        # Theorem 2.3, equation (2.9)
        if subtract_median:
            median = 2*np.log( tau*C_dk*SIGMA_A ) - 2*np.log(np.log(2)) + 2*np.log( 1 + (k-2+ 1/d)**2 * np.log(np.log(n))/((1-1/d)*np.log(n)) )
            # The above is the median of the corrected cdf
            beta = beta + median
        correction = C_dk*tau*SIGMA_A*np.exp(-0.5*beta)*(k-2 + 1/d)**2 * np.log(np.log(n)) / ( (1 - 1/d)*np.log(n))
    # Note that if subtract_median = True,
    # then we have already modified beta accordingly, and so do not
    # need to do it again in limit().
    return np.exp(-correction)*limit(beta,tau,d,k,subtract_median=False)

def compute_gamma(beta, t,d,k):
    """
    Numerically computes gamma_t for the gamma defined just before
    Lemma 4.1
    """
    theta_d   = theta(d)
    theta_dm1 = theta(d-1)
    f0 = 1/theta_d
    if d == 2 and k == 1:
        total_volume = np.log(t) + beta
    else:
        total_volume = (2-2/d)*np.log(t) - (4 -2*k - 2/d)*np.log(np.log(t)) + beta
    rt = (total_volume/(t*theta_d*f0))**(1/d)
    def G(u):
        return quad(lambda s : (1 - s*s)**((d-1)/2),-1, u)[0]
    rtd = rt**d
    rtsq= rt*rt
    p0 = poisson.cdf(k-1,total_volume)
    def p(alpha):
        """
        Using the parameterisation R = 1 - alpha r_t
        for alpha in [0,1], we can compute the probability
        that (R,0,...,0) is not k-covered.
        """
        left_crescent_limit = (alpha - 0.5*(alpha*alpha + 1)*rt)/(1 - alpha*rt)
        delta = 0.5*(1 - alpha*alpha)/(1 - alpha*rt)
        left_contrib = G(left_crescent_limit)
        right_contrib = rt* delta**((d+1)/2) * quad(lambda s : (s*(2 - delta*rtsq*s))**((d-1)/2),0,1)[0]
        rate = (theta_dm1/theta_d) * total_volume * (left_contrib + right_contrib)
        return poisson.cdf(k-1,rate)
    bulk_term     = t*f0*theta_d*p0*(1-rt)**d
    boundary_term = t*f0*d*theta_d*rt*quad( lambda alpha : (1-alpha*rt)**(d-1)*p(alpha), 0, 1 )[0]
    return bulk_term + boundary_term

def lhs_quantity( R, n, k, d ):
    """
    Given R, calculates the random variable whose limit we consider in
    Theorem 2.1, 2.2 and 2.3.
    Recall that everything in this file takes place inside a unit ball,
    so theta_d * f_0 = 1.
    """
    #THETA_d = theta(d)
    #f0 = 1/THETA_d
    if d==2 and k == 1:
        return n*np.square(R) - np.log(n)
    else:
        return n*np.power(R,d) - (2 - 2/d)*np.log(n) - (2*k - 4 + 2/d)*np.log(np.log(n))

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
            progress.set_description("Step 1/4: sampling n points")
            Xn = sample_point(d, n)
            progress.set_description("Step 2/4: building k-d tree")
            tree = KDTree(Xn)
            progress.set_description("Step 3/4: sampling m points")
            Yn = sample_point(d, m)
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

if __name__=='__main__':
    # Arguments for the script are: n, tau, d, k, batch_size
    #Constants:
    n   = int(sys.argv[1])
    tau = int(sys.argv[2])
    m = int(n*tau)
    domain = "ball" # Redundant
    d = int(sys.argv[3])
    k = int(sys.argv[4])
    if len(sys.argv) >= 6:
        batch_size = int(sys.argv[5])
    else:
        batch_size = 2 # The number of times we will sample R_{n,m}.
    p = 0.1
    # The file contains transformed data, i.e. we've already applied lhs_quantity.
    filename = f'{d}d-{domain}/n{n}-tau{tau}-k{k}.csv'
    
    # subtract_median_diagram(n,tau,d,k,filename=filename,outname=f'median/median-test-{d}d-{domain}-n{n}-tau{tau}-k{k}.png')
    
    # subtract_median_diagram_corrected(n,tau,d,k,filename=filename,outname=f'median/median-test-{d}d-{domain}-n{n}-tau{tau}-k{k}-corrected-limit.png')
    
    # ten_percentiles(n,tau,d,k,batch_size,filename=filename,abs_tolerance=p)
    
    # diagram_with_corrected_limit(n,tau,d,k,filename=filename,outname=f'diagrams/{d}d-{domain}-with-fluctuation-n{n}-k{k}-tau{tau}.png')
    # diagram_with_numerical_computation(n,tau,d,k,filename=filename,outname=f'{d}d-{domain}-with-fluctuation-n{n}-k{k}-tau{tau}-nc.png')

