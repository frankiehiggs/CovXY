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
    return 2*dim

@jit(nopython=True,parallel=False)
def sample_point( dim, sample_size ):
    """
    Chooses a point uniformly in [0,1]^d.
    """
    return np.random.random(size=(sample_size,dim))

def limit( beta, tau, dim, k ):
    """
    Evaluates the limiting cdfs from Theorem 2.4 and Theorem 2.6
    of the August 30 version.
    This is not the limiting cdf of R_{n,m} itself,
    but of the derived quantity n theta(dim) f_0 R^d - log...
    """
    return np.exp( - tau*np.exp(-beta) / np.math.factorial(k-1) )

def corrected_limit(beta, tau, dim, k, n):
    """
    Returns the "corrected limit" from Theorem 2.4 and Theorem 2.6.
    I've used exp(-x) instead of 1-x to ensure that what we get
    is a cdf.
    """
    # NOTE
    # This isn't for the torus, I need to modify it.
    SIGMA_A = sigma_A(dim)
    C_dk = c_dk(dim,k)
    correction = tau * np.exp(-beta) * (k-1)*(k-1) * np.log(np.log(n)) / ( np.math.factorial(k-1) * np.log(n) )
    return np.exp(-correction)*limit(beta,tau,dim,k)

@jit(nopython=True)
def shift(torus_points,coords):
    shifted = torus_points.copy()
    for i in range(torus_points.shape[0]):
        for coord in coords:
            shifted[i,coord] = (shifted[i,coord] + 0.5) % 1.0
    return shifted

def compute_gamma(beta, t,dim,k):
    """
    Numerically computes gamma_t for the gamma defined just before
    Lemma 3.3 (in the August 30 version).
    """
    theta_d   = theta(dim)
    theta_dm1 = theta(dim-1)
    f0 = 1/theta_d
    if dim == 2 and k == 1:
        total_volume = np.log(t) + beta
    else:
        total_volume = (2-2/dim)*np.log(t) - (4 -2*k - 2/dim)*np.log(np.log(t)) + beta
    rt = (total_volume/(t*theta_d*f0))**(1/dim)
    def G(u):
        return quad(lambda s : (1 - s*s)**((dim-1)/2),-1, u)[0]
    rtd = rt**dim
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
        right_contrib = rt* delta**((dim+1)/2) * quad(lambda s : (s*(2 - delta*rtsq*s))**((dim-1)/2),0,1)[0]
        rate = (theta_dm1/theta_d) * total_volume * (left_contrib + right_contrib)
        return poisson.cdf(k-1,rate)
    bulk_term     = t*f0*theta_d*p0*(1-rt)**dim
    boundary_term = t*f0*dim*theta_d*rt*quad( lambda alpha : (1-alpha*rt)**(dim-1)*p(alpha), 0, 1 )[0]
    return bulk_term + boundary_term

def lhs_quantity( R, n, k, dim ):
    """
    Given R, calculates the random variable whose limit we consider in
    Theorem 2.4 and Theorem 2.6 (of the August 30 version).
    Recall that everything in this code takes place inside a unit ball,
    so theta_d * f_0 = 1.
    """
    THETA_d = theta(dim)
    f0 = 1
    if k == 1:
        return n*THETA_d*f0*R**dim - np.log(n)
    else:
        return n*THETA_d*f0*R**dim - np.log(n) - (k-1)*np.log(np.log(n))

@jit(nopython=True)
def generate_Rk(n,m,dim,k):
    Xn = np.random.random(size=(n,dim))
    Ym = np.random.random(size=(m,dim))
    k_nearest_dists = np.empty(m)
    for j, y in enumerate(Ym):
        closest_k = dim*np.ones(k)
        for x in Xn:
            too_far = False
            total_distance = 0
            for i in range(dim):
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
def generate_R_samples(n, m, dim, k, number_of_samples=2):
    """
    Produces samples of the coverage threshold R_{n,m}.
    This function takes up the majority of the runtime.
    """
    samples = np.empty(number_of_samples)
    progress = tqdm(range(number_of_samples))
    if k == 1:
        for s in progress:
            Xn = sample_point(dim, n)
            tree_0 = KDTree(Xn)
            Yn = sample_point(dim, m)
            # Measure max_j min_i d(Y_j, X_i):
            distances = tree_0.query(Yn)[0]
            for r in range(1,dim+1):
                for choice in combinations(range(dim),r):
                    shifted_Xn   = shift(Xn,choice)
                    shifted_tree = KDTree(shifted_Xn)
                    shifted_Yn   = shift(Yn,choice)
                    distances = np.minimum(distances, shifted_tree.query(shifted_Yn)[0])
            samples[s] = np.max( distances )
    else:
        for s in progress:
            samples[s] = generate_Rk(n,m,dim,k)
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

def compute_quantiles( required_quantiles,abs_tolerance,rel_tolerance,confidence, n,m,dim,k, batch_size = 2,filename=None ):
    """
    Given a list of required quantiles [p1, ..., pk],
    estimates each quantile to within the requested absolute and relative tolerances,
    and with the given confidence level.
    Algorithm from "How to estimate quantiles easily and reliably",
    Keith Briggs and Fabian Ying.
    
    If there is pre-existing data at filename, it will be loaded,
    and the algorithm will produce extra samples as needed.
    If filename was given, any new data generated will be appended to the file.
    """
    print("Generating or loading initial samples")
    if filename:
        try:
            samples = np.genfromtxt(filename)
        except:
            samples = lhs_quantity(generate_R_samples(n, m, dim, k, batch_size),n,k,dim)
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
        print(f'\nContinuing the n={n:.0e}, m={m:.0e}, d={dim}, k={k} simulation in the torus...')
        # If we've not met the tolerances then we generate more samples,
        # then test again.
        print(f'\nGenerating new samples! This is attempt number {attempts}.')
        attempts += 1
        new_samples = lhs_quantity(generate_R_samples(n, m, dim, k, batch_size),n,k,dim)
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

def ten_percentiles(n,tau,dim,k,batch_size=2,abs_tolerance=0.1,rel_tolerance=0.1,confidence=0.95,filename=None):
    """
    Computes the 10th, 20th, ..., 90th percentiles
    to the given tolerance and confidence.
    """
    m = int(n*tau)
    desired_quantiles = np.linspace(start=0.1,stop=0.9,num=9)
    quantiles = compute_quantiles(desired_quantiles, abs_tolerance,rel_tolerance,confidence, n,m,dim,k, batch_size, filename)
    return quantiles

def diagram_with_limit(n, tau, dim, k, filename=None, samples=None,outname=None):
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
    p_limit = limit(my_range,tau,dim,k)
    ax.plot(my_range,p_limit,'k--',linewidth=1.5,label="Limiting cdf")
    
    ax.set_ylim(0,1)
    ax.set_xlim(min(samples)-0.1,6)
    ax.legend(loc='lower right')
    ax.set_title(f'n=$10^{int(np.log10(n))}$, tau={tau} and k={k} on a {dim}-dimensional torus.')
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    
    if outname:
        fig.savefig(outname)
    else:
        plt.show()
    plt.close()

def diagram_with_corrected_limit(n,tau,dim,k,filename=None, samples=None,outname=None):
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
    print('Histogram computed!\n')
    
    my_range = np.arange(min(samples)-0.1,max(samples)+0.1,0.01)
    p_limit = limit(my_range,tau,dim,k)
    ax.plot(my_range,p_limit,'k--',linewidth=2,label="Limiting cdf")
    p_corr = corrected_limit(my_range, tau, dim, k, n)
    ax.plot(my_range,p_corr ,'r',linestyle='dotted',linewidth=3,label="Corrected cdf from Theorem NUMBER")

    ax.set_ylim(0,1)
    ax.set_xlim(min(samples)-0.1, 6)
    ax.legend(loc='lower right')
    ax.set_title(f'n=10^{int(np.log10(n))}, tau={tau}, d={dim} and k={k}.')
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    
    if outname:
        fig.savefig(outname)
    else:
        plt.show()
    plt.close()
    
def empirical_cdf_diagram(n, tau, dim, k, filename=None, samples=None,outname=None):
    """
    Plots the empirical cdf.
    Requires the generated list of samples as input.
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
    
    ax.set_ylim(0,1)
    ax.set_xlim(min(samples)-0.1,max(samples)+0.1)
    ax.legend(loc='lower right')
    ax.set_title(f'n={n}, tau={tau} and k={k} on a {dim}-dimensional torus.')
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    
    if outname:
        fig.savefig(outname)
    else:
        plt.show()
    plt.close()

if __name__=='__main__':
    # Arguments for the script are: n, tau, dim, k, batch_size, p
    #Constants:
    n   = int(sys.argv[1])
    tau = int(sys.argv[2])
    m = int(n*tau)
    dim = int(sys.argv[3])
    k = int(sys.argv[4])
    if len(sys.argv) >= 6:
        batch_size = int(sys.argv[5])
    else:
        batch_size = 2 # The number of times we will sample R_{n,m}.
    if len(sys.argv) >= 7:
        p = float(sys.argv[6])
    else:
        p = 0.1
    # The file contains transformed data, i.e. we've already applied lhs_quantity.
    filename = f'torus/data/d{dim}-k{k}-tau{tau}-n{n}.csv'
    imgname  = f'torus/diagrams/d{dim}-k{k}-tau{tau}-n{n}-w-lim.png'
    
    ten_percentiles(n,tau,dim,k,batch_size,filename=filename,abs_tolerance=p)
    
    diagram_with_limit(n,tau,dim,k,filename=filename,outname=imgname)

