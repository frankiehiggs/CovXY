"""
September 1 update: I've removed the domain argument:
everything is done inside a ball now.
k > 1 is more fully implemented.
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
def theta(dim):
    """
    Returns the volume of the d-dimensional unit ball.
    """
    return np.pi**(dim/2) / np.math.gamma(dim/2 + 1)

def c_dk(dim,k):
    if k==1:
        if dim==2:
            return 1
        else:
            return (1/theta(dim-1)) * (theta(dim)/(2 - 2/dim))**(1 - 1/dim)
    else:
        return (theta(dim)**(1 - 1/dim) * (1 - 1/dim)**(k-2+1/dim))/(np.math.factorial(k-1)*2**(1-1/dim)*theta(dim-1))

def sigma_A(dim):
    return 4

@jit(nopython=True,parallel=False)
def sample_point( sample_size ):
    """
    Chooses a point uniformly in the square [0,1]^2.
    """
    return np.random.random(size=(sample_size,2))

# def rn( beta, dim, n ):
    # """
    # Returns r_n given the relevant quantities.
    # Only usable in the k=1 case for now.
    # """
    # THETA_d = theta(dim)
    # f0 = 1/THETA_d
    # big_term = (2 - 2/dim) * (np.log(n) - np.log(np.log(n)))
    # return np.power( (big_term + beta)/(n*f0*THETA_d) , 1/dim)

def limit( beta, tau ):
    """
    This is not the limiting cdf of R_{n,m} itself,
    but of the derived quantity n theta(dim) f_0 R^2 - log...
    """
    return np.exp( - tau*np.exp(-beta) )

def corrected_limit(beta, tau, n):
    """
    Returns the "corrected limit" from Theorem 2.4 and Theorem 2.6.
    I've used exp(-x) instead of 1-x to ensure that what we get
    is a cdf.
    """
    SIGMA_A = 4
    correction = tau * np.sqrt(np.pi) * SIGMA_A * np.exp(-0.5*beta) / (2 * np.sqrt(np.log(n)))
    return np.exp(-correction)*limit(beta,tau)

def lhs_quantity( R, n ):
    """
    Given R, calculates the random variable whose limit we consider in
    Theorem 2.4 and Theorem 2.6 (of the August 30 version).
    Recall that everything in this code takes place inside a unit ball,
    so theta_d * f_0 = 1.
    """
    f0 = 1
    return n*np.pi*f0*np.square(R) - np.log(n)

@jit
def generate_R_samples(n, m, number_of_samples=2):
    """
    Produces samples of the coverage threshold R_{n,m}.
    This function takes up the majority of the runtime.
    """
    samples = np.empty(number_of_samples)
    progress = tqdm(range(number_of_samples))
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
    return samples

def save_data( filename, samples ):
    f = open(filename, 'a')
    #print("Writing data to file...")
    for s in samples:
        f.write(str(s)+'\n')
    #print("Saved!")

def find_rs(p, sample_size, confidence, exact=True):
    """
    Finds r and s with s-r minimal
    such that P( r <= N < s ) > confidence,
    where N is a Binomial(p, max_index+1) random variable.
    
    Note on indexing: I've written most of this function to be consistent
    with Briggs' and Ying's notation,
    so r and s are in [1, n] rather than [0,n-1] until the return statement,
    then we subtract 1 from each before returning.
    """
    r,s = binom.interval(confidence,sample_size,p)
    return (max(int(r)-1,0), int(s)-1)
    
def insert_new_elements( old_list, new_elements ):
    """
    Given two sorted numpy arrays, creates their sorted union
    by inserting the elements of the second array into the first.
    It is assumed that both old_list and new_elements are sorted.
    """
    merged_list = np.empty(shape=(old_list.size + new_elements.size))
    i = 0
    j = 0
    while i < old_list.size and j < new_elements.size:
        if old_list[i] < new_elements[j]:
            merged_list[i+j] = old_list[i]
            i += 1
        else:
            merged_list[i+j] = new_elements[j]
            j += 1
    if (i == old_list.size):
        merged_list[i+j:] = new_elements[j:]
    else:
        merged_list[i+j:] = old_list[i:]
    return merged_list
    
def meets_tolerances(samples, pairs_list, abs_tolerance,rel_tolerance):
    """
    Checks if the tolerance conditions are met.
    pairs_list is a list of the pairs (r_k, s_k).
    samples is assumed to be sorted.
    """
    width = samples[-1] - samples[0]
    tolerance = min(abs_tolerance, rel_tolerance*width)
    for i,pair in enumerate(pairs_list):
        r,s = pair
        if (samples[s] - samples[r] > tolerance):
            print(f'\nQuantile {i+1} has c.i. ({samples[r]:.3f},{samples[s]:.3f}), width {samples[s]-samples[r]:.4f}, but the tolerance is {tolerance:.4f}')
            return False
    return True

def compute_quantiles( required_quantiles,abs_tolerance,rel_tolerance,confidence, n,m, batch_size = 2,filename=None ):
    """
    Given a list of required quantiles [p1, ..., pk],
    estimates each quantile to within the requested absolute and relative tolerances,
    and with the given confidence level.
    Algorithm from "How to estimate quantiles easily and reliably",
    Keith Briggs and Fabian Yang.
    
    If there is pre-existing data at filename, it will be loaded,
    and the algorithm will produce extra samples as needed.
    If filename was given, any new data generated will be appended to the file.
    """
    print("Generating or loading initial samples")
    if filename:
        try:
            samples = np.genfromtxt(filename)
        except:
            samples = lhs_quantity(generate_R_samples(n, m, batch_size),n)
            save_data(filename, samples)
    else:
        raise Exception("Please provide a filename for the data (even if the data file doesn't exist yet).")
    print("Sorting initial samples")
    samples.sort()
    N = samples.size
    # Create the (r,s) pairs and check if the tolerance is met.
    pairs_list = [(0,0)]*len(required_quantiles)
    print("Finding (r,s) pairs")
    for i,p in enumerate(required_quantiles):
        pairs_list[i] = find_rs(p, N, confidence,exact=False)
    attempts = 1
    while not meets_tolerances(samples, pairs_list, abs_tolerance,rel_tolerance):
        if attempts % 10 == 0:
            print("Loading some new samples")
            samples = np.genfromtxt(filename)
            N = samples.size
            samples.sort()
        print(f'\nContinuing the n={n:.0e}, m={m:.0e}, simulation in the {"unit square"}...')
        # If we've not met the tolerances then we generate more samples,
        # then test again.
        print(f'\nGenerating new samples! This is attempt number {attempts}.')
        attempts += 1
        new_samples = lhs_quantity(generate_R_samples(n, m,  batch_size),n)
        if filename:
            save_data(filename, new_samples)
        new_samples.sort()
        samples = insert_new_elements(samples, new_samples)
        N += batch_size
        for i,p in enumerate(required_quantiles):
            pairs_list[i] = find_rs(p,N,confidence,exact=False)
    quantiles = np.empty(shape=(len(required_quantiles)))
    for i, p in enumerate(required_quantiles):
        quantiles[i] = samples[int(p*N)-1]
    return quantiles

def ten_percentiles(n,tau,batch_size=2,abs_tolerance=0.1,rel_tolerance=0.1,confidence=0.95,filename=None):
    """
    Computes the 10th, 20th, ..., 90th percentiles
    to the given tolerance and confidence.
    """
    m = int(n*tau)
    desired_quantiles = np.linspace(start=0.1,stop=0.9,num=9)
    quantiles = compute_quantiles(desired_quantiles, abs_tolerance,rel_tolerance,confidence, n,m, batch_size, filename)
    return quantiles

def diagram_with_limit(n, tau, filename=None, samples=None,outname=None):
    """
    Plots the limiting cdf and empirical cdf on the same axes. Requires the generated list of samples as input.
    """
    fig, ax = plt.subplots()
    if filename:
        print("Loading samples...")
        samples = np.genfromtxt(filename)
    elif not samples:
        raise Exception("Please input some data, either as a filename or numpy array")
    print("Computing histogram...")
    ax.hist(samples, 10000, density=True, cumulative=True, histtype='step', label="Empirical", log=False)
    print('Histogram computed!\n')
    
    my_range = np.arange(min(samples)-0.1,max(samples)+0.1,0.01)
    p_limit = limit(my_range,tau)
    ax.plot(my_range,p_limit,'k--',linewidth=1.5,label="Limiting cdf")
    
    ax.set_ylim(0,1)
    ax.set_xlim(-2,15)
    ax.legend(loc='lower right')
    ax.set_title(f'n=$10^{int(np.log10(n))}$, tau={tau} in the square $[0,1]^2$.')
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    
    if outname:
        fig.savefig(outname)
    else:
        plt.show()
    plt.close()

def diagram_with_corrected_limit(n,tau,filename=None, samples=None,outname=None):
    """
    Creates a figure showing the empirical cdf,
    limiting cdf, and the cdf containing the first-order terms for both the boundary and bulk.
    """
    fig, ax = plt.subplots()
    if filename:
        print("Loading samples...")
        samples = np.genfromtxt(filename)
    elif not samples:
        print("Please input some data, either as a filename or numpy array")
        return
    print("Computing histogram...")
    ax.hist(samples, 10000, density=True, cumulative=True, histtype='step', label="Empirical", log=False, linewidth=2)
    # New method from Keith: sort list and plot (scaled) indices against the sorted list.
    print('Histogram computed!\n')

    my_range = np.arange(min(samples)-0.1,max(samples)+0.1,0.01)
    p_limit = limit(my_range,tau)
    ax.plot(my_range,p_limit,'k--',linewidth=2,label="Limiting cdf")
    p_corr = corrected_limit(my_range, tau, n)
    ax.plot(my_range,p_corr ,'r',linestyle='dotted',linewidth=3,label="Corrected cdf from Theorem NUMBER")
    
    ax.set_ylim(0,1)
    ax.set_xlim(-2,15)
    ax.legend(loc='lower right')
    ax.set_title(f'$n=10^{int(np.log10(n))}$, tau={tau} in the unit square $[0,1]^2$.')
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    
    if outname:
        fig.savefig(outname)
    else:
        plt.show()
    plt.close()
    
if __name__=='__main__':
    # Arguments for the script are: n, tau, dim, k, batch_size
    #Constants:
    n   = int(sys.argv[1])
    tau = int(sys.argv[2])
    m = int(n*tau)
    if len(sys.argv) >= 4:
        batch_size = int(sys.argv[3])
    else:
        batch_size = 100 # The number of times we will sample R_{n,m}.
    p = 0.1
    # The file contains transformed data, i.e. we've already applied lhs_quantity.
    filename = f'square/n{n}-tau{tau}.csv'
    
    ten_percentiles(n,tau,batch_size,filename=filename,abs_tolerance=p)
    
    diagram_with_corrected_limit(n,tau,filename=filename,outname=f'square/square-n{n}-tau{tau}.png')
