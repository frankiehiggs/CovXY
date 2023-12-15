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
        return (theta(dim)**(1 - 1/dim) * (1 - 1/dim)**(k-2+ 1/dim))/(np.math.factorial(k-1)*2**(1-1/dim) * theta(dim-1))

def sigma_A(dim):
    return dim * (theta(dim)**(1/dim))

@jit(nopython=True,parallel=False)
def sample_point( dim, sample_size ):
    """
    Chooses a point uniformly in the unit ball.
    """
    if dim==2:
        theta = 2*np.pi*np.random.random(size=(sample_size,1))
        radius_sqrt = np.sqrt(np.random.random(size=(sample_size,1)))
        x = radius_sqrt * np.cos(theta)
        y = radius_sqrt * np.sin(theta)
        samples = np.concatenate((x,y),axis=1)
    elif dim<=4:
        # If d isn't too large then the fastest way to sample a point in the unit ball
        # is to repeatedly sample a point in the box [-1,1]^d until we get one which is
        # until we get a point inside the unit disc.
        samples = 2*np.random.random(size=(sample_size,dim)) - 1
        for i in range(sample_size):
            while np.dot(samples[i,:],samples[i,:]) > 1:
                samples[i,:] = 2*np.random.random(size=dim) - 1
    else:
        # uniform sample in the (solid) d-dimensional unit ball
        # Using "polar coordinates":
        # We choose a point uniformly on the unit sphere,
        # then multiply it by U^{1/d},
        # where U ~ U[0,1].
        samples = np.empty(shape=(sample_size,dim)) # Each column is a Gaussian rv in d dimensions
        for i in range(sample_size):
            row = np.empty(dim)
            for j in range(dim):
                row[j] = np.random.normal()
            row /= np.linalg.norm(row)
            row *= np.random.random()**(1/dim)
            samples[i,:] = row
    return samples

# def rn( beta, dim, n ):
    # """
    # Returns r_n given the relevant quantities.
    # Only usable in the k=1 case for now.
    # """
    # THETA_d = theta(dim)
    # f0 = 1/THETA_d
    # big_term = (2 - 2/dim) * (np.log(n) - np.log(np.log(n)))
    # return np.power( (big_term + beta)/(n*f0*THETA_d) , 1/dim)

def limit( beta, tau, dim, k, subtract_median = False ):
    """
    Evaluates the limiting cdfs from Theorem 2.4 and Theorem 2.6
    of the August 30 version.
    This is not the limiting cdf of R_{n,m} itself,
    but of the derived quantity n theta(dim) f_0 R^d - log...
    """
    SIGMA_A = sigma_A(dim)
    C_dk = c_dk(dim,k)
    if dim==2:
        if k == 1:
            if subtract_median:
                beta = beta + np.log(tau) - np.log(np.log(2))
            return np.exp(-tau*np.exp(-beta))
        elif k == 2:
            if subtract_median:
                median = -2*np.log( -0.125*np.sqrt(np.pi)*SIGMA_A + np.sqrt(np.pi*SIGMA_A*SIGMA_A/64 + np.log(2)/tau) )
                beta = beta + median
            return np.exp(-tau*( np.exp(-beta) + 0.25*np.sqrt(np.pi)*SIGMA_A*np.exp(-0.5*beta) ) )
        else:
            if subtract_median:
                median = 2*np.log( tau*SIGMA_A*np.sqrt(np.pi)*(0.5**k) / (np.math.factorial(k-1)*np.log(2)) )
                beta = beta + median
            return np.exp( -tau*SIGMA_A*np.exp(-0.5*beta)*np.sqrt(np.pi)*(0.5**k) / np.math.factorial(k-1)  )
    else:
        if subtract_median:
            median = 2*np.log( tau * C_dk * SIGMA_A / np.log(2) )
            beta = beta + median
        return np.exp( -tau * C_dk * SIGMA_A * np.exp(-beta/2) )

def corrected_limit(beta, tau, dim, k, n,subtract_median=False):
    """
    Returns the "corrected limit" from Theorem 2.4 and Theorem 2.6.
    I've used exp(-x) instead of 1-x to ensure that what we get
    is a cdf.
    """
    SIGMA_A = sigma_A(dim)
    C_dk = c_dk(dim,k)
    if dim == 2:
        if k == 1:
            if subtract_median:
                median = -2*np.log(-0.25*np.sqrt(np.pi/np.log(n))*SIGMA_A + np.sqrt(np.pi*SIGMA_A*SIGMA_A / (16*np.log(n)) + np.log(2)/tau))
                beta = beta + median
            correction = 0.5*np.sqrt(np.pi)*tau*SIGMA_A*(np.log(n))**(-0.5)*np.exp(-0.5*beta)
            # NOTE this is the correct expression from the proof
            # of Theorem 2.4 in Section 5, not the erroneous formula
            # in the statement of Theorem 2.4.
            #
            # I still need to check all the other cases
            # have been stated correctly. They don't look too good,
            # but that might be from missing out the bulk terms.
        elif k == 2:
            correction = tau*np.sqrt(np.pi)*SIGMA_A*np.log(np.log(n))*np.exp(-0.5*beta)/(8*np.log(n))
        else:
            #correction = tau*SIGMA_A*np.exp(-0.5*beta)*np.sqrt(np.pi)*(2*k - 3)**2 *np.log(np.log(n)) / ( np.math.factorial(k-1)*2**(k+1)*np.log(n) )
            correction = C_dk * tau * SIGMA_A * np.exp(-0.5*beta) * (k - 2 + 1/dim)**2 * np.log(np.log(n)) / ( (1 - 1/dim) * np.log(n) )
    else:
        # correction = tau*SIGMA_A*C_dk*np.exp(-0.5*beta)*( 2*np.log(np.log(n))*(k-2+1/dim)**2 + (k-2+1/dim)*beta + (2*k-2)/(1-1/dim) ) / np.log(n)
        # # The above is the correction with some extra terms in it from
        # # Proposition 5.2.
        # # Personally I'm not so happy with the loose beta
        # # there; it might make the "corrected cdf" not a cdf.
        correction = C_dk * tau * SIGMA_A * np.exp(-0.5*beta) * (k - 2 + 1/dim)**2 * np.log(np.log(n)) / ( (1 - 1/dim) * np.log(n) )
        # The above is just the corrected cdf from the statement of Theorem 2.3.
    return np.exp(-correction)*limit(beta,tau,dim,k)

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
    f0 = 1/THETA_d
    if dim==2 and k == 1:
        return n*THETA_d*f0*np.square(R) - np.log(n)
    else:
        THETA_d = theta(dim)
        f0 = 1/THETA_d
        return n*THETA_d*f0*np.power(R,dim) - (2 - 2/dim)*np.log(n) - (2*k - 4 + 2/dim)*np.log(np.log(n))

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
            progress.set_description("Step 1/4: sampling n points")
            Xn = sample_point(dim, n)
            progress.set_description("Step 2/4: building k-d tree")
            tree = KDTree(Xn)
            progress.set_description("Step 3/4: sampling m points")
            Yn = sample_point(dim, m)
            # Measure max_j min_i d(Y_j, X_i):
            progress.set_description("Step 4/4: measure distances")
            distances = tree.query(Yn)[0]
            samples[s] = np.max( distances )
    else:
        for s in progress:
            progress.set_description("Step 1/4: sampling n points")
            Xn = sample_point(dim, n)
            progress.set_description("Step 2/4: building k-d tree")
            tree = KDTree(Xn)
            progress.set_description("Step 3/4: sampling m points")
            Yn = sample_point(dim, m)
            progress.set_description("Step 4/4: measure distances")
            distances = tree.query(Yn,k=k)[0][:,k-1]
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

def compute_quantiles( required_quantiles,abs_tolerance,rel_tolerance,confidence, n,m,dim,k, batch_size = 2,filename=None ):
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
        print(f'\nContinuing the n={n:.0e}, m={m:.0e}, d={dim}, k={k} simulation in the {"ball"}...')
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
    ax.set_xlim(-2,15)
    ax.legend(loc='lower right')
    ax.set_title(f'n=$10^{int(np.log10(n))}$, tau={tau}, d={dim} and k={k}.')
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
    # New method from Keith: sort list and plot (scaled) indices against the sorted list.
    print('Histogram computed!\n')

    my_range = np.arange(min(samples)-0.1,max(samples)+0.1,0.01)
    p_limit = limit(my_range,tau,dim,k)
    ax.plot(my_range,p_limit,'k--',linewidth=2,label="Limiting cdf")
    p_corr = corrected_limit(my_range, tau, dim, k, n)
    ax.plot(my_range,p_corr ,'r',linestyle='dotted',linewidth=3,label="Corrected cdf from Theorem NUMBER")
    
    ax.set_ylim(0,1)
    ax.set_xlim(-2,15)
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
    
def diagram_with_numerical_computation(n,tau,dim,k,filename=None, samples=None,outname=None):
    """
    Creates a figure showing the empirical cdf,
    limiting cdf, and exp(-tau h_n(beta,n)),
    where h_n = n f_0 \int_A p(x)dx is computed numerically.
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
    print("Histogram computed!")

    my_range = np.arange(min(samples)-0.1,max(samples)+0.1,0.05)
    p_limit = limit(my_range,tau,dim,k)
    ax.plot(my_range,p_limit,'k--',linewidth=2,label="Limiting cdf")
    print('Computing numerical estimate for the cdf (can be slow...)\n')
    p_num = np.exp(-tau*np.vectorize(compute_gamma)(my_range, n, dim, k))
    ax.plot(my_range,p_num ,'g',linestyle='dotted',linewidth=3,label="$e^{-\\tau \\gamma_n}$")
    
    ax.set_ylim(0,1)
    ax.set_xlim(-2,15)
    ax.legend(loc='lower right')
    ax.set_title(f'n={n}, tau={tau}, d={dim} and k={k}')
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout
    
    if outname:
        fig.savefig(outname)
    else:
        plt.show()
    plt.close()
    
def subtract_median_diagram(n,tau,dim,k,filename=None, samples=None,outname=None):
    """
    Does something, which I'll describe later.
    """
    fig, ax = plt.subplots()
    if filename:
        print("Loading samples...")
        samples = np.genfromtxt(filename)
    elif not samples:
        print("Please input some data, either as a filename or numpy array")
        return
    print("Computing histogram...")
    samples -= np.median(samples)
    ax.hist(samples, 10000, density=True, cumulative=True, histtype='step', label="Empirical", log=False, linewidth=2)
    print("Histogram computed!")

    my_range = np.arange(min(samples)-0.1,max(samples)+0.1,0.05)
    p_limit = limit(my_range,tau,dim,k,subtract_median=True)
    ax.plot(my_range,p_limit,'k--',linewidth=2,label="Limiting cdf")
    
    ax.set_ylim(0,1)
    ax.set_xlim(-10,20)
    ax.legend(loc='lower right')
    ax.set_title(f'n={n}, tau={tau}, d={dim} and k={k}, with the medians subtracted')
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout
    
    if outname:
        fig.savefig(outname)
    else:
        plt.show()
    plt.close()
    
def subtract_median_diagram_corrected(n,tau,dim,k,filename=None, samples=None,outname=None):
    """
    Subtracts the median from the corrected limit as well.
    Currently only implemented for d=2, k=1.
    I need to modify corrected_limit for the other parameters.
    """
    fig, ax = plt.subplots()
    if filename:
        print("Loading samples...")
        samples = np.genfromtxt(filename)
    elif not samples:
        print("Please input some data, either as a filename or numpy array")
        return
    print("Computing histogram...")
    samples = samples - np.median(samples)
    ax.hist(samples, 10000, density=True, cumulative=True, histtype='step', label="Empirical", log=False, linewidth=2)
    print('Histogram computed!\n')

    my_range = np.arange(min(samples)-0.1,max(samples)+0.1,0.01)
    p_limit = limit(my_range,tau,dim,k,subtract_median=True)
    ax.plot(my_range,p_limit,'k--',linewidth=2,label="Limiting cdf")
    p_corr = corrected_limit(my_range, tau, dim, k, n,subtract_median=True)
    ax.plot(my_range,p_corr ,'r',linestyle='dotted',linewidth=3,label="Corrected cdf from Theorem NUMBER")
    
    ax.set_ylim(0,1)
    ax.set_xlim(-10,20)
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

if __name__=='__main__':
    # Arguments for the script are: n, tau, dim, k, batch_size
    #Constants:
    n   = int(sys.argv[1])
    tau = int(sys.argv[2])
    m = int(n*tau)
    domain = "ball" # Redundant
    dim = int(sys.argv[3])
    k = int(sys.argv[4])
    if len(sys.argv) >= 6:
        batch_size = int(sys.argv[5])
    else:
        batch_size = 2 # The number of times we will sample R_{n,m}.
    p = 0.1
    # The file contains transformed data, i.e. we've already applied lhs_quantity.
    filename = f'{dim}d-{domain}/n{n}-tau{tau}-k{k}.csv'
    
    # subtract_median_diagram(n,tau,dim,k,filename=filename,outname=f'median/median-test-{dim}d-{domain}-n{n}-tau{tau}-k{k}.png')
    
    # subtract_median_diagram_corrected(n,tau,dim,k,filename=filename,outname=f'median/median-test-{dim}d-{domain}-n{n}-tau{tau}-k{k}-corrected-limit.png')
    
    # ten_percentiles(n,tau,dim,k,batch_size,filename=filename,abs_tolerance=p)
    
    # diagram_with_corrected_limit(n,tau,dim,k,filename=filename,outname=f'diagrams/{dim}d-{domain}-with-fluctuation-n{n}-k{k}-tau{tau}.png')
    # diagram_with_numerical_computation(n,tau,dim,k,filename=filename,outname=f'{dim}d-{domain}-with-fluctuation-n{n}-k{k}-tau{tau}-nc.png')

