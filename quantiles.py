"""
Implements the method of Briggs and Ying for finding a stopping time
(as we continue to collect samples)
when we have a sufficiently good estimate for certain quantiles.

See their article, available at
https://ora.ox.ac.uk/objects/uuid:4cd0c80b-6d7b-41f5-a4f0-e5dd0cd2515b 
"""
import numpy as np
from scipy.stats import binom

def save_data( filename, samples ):
    f = open(filename, 'a')
    for s in samples:
        f.write(str(s)+'\n')

def find_rs(p, sample_size, confidence):
    """
    Finds r and s with s-r minimal
    such that P( r <= N < s ) > confidence,
    where N is a Binomial(p, max_index+1) random variable.
    
    Note on indexing: I've written most of this function to be consistent
    with the notation in Briggs' and Ying's article,
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

def compute_quantiles(required_quantiles,abs_tolerance,rel_tolerance,confidence,filename,sampling_function,*model_params):
    """
    Given a list of required quantiles [p1, ..., pk],
    estimates each quantile to within the requested absolute and relative tolerances,
    and with the given confidence level.
    
    model_params are the arguments to sampling_function,
    which should be the function returning samples of
    n f_0 theta_d R_{n,m,k}^d - c_1 log(n) - c_2 loglog(n)
    for whichever setting we are considering.
    The parameters are usually n, m, d, k and the batch size.
    """
    try:
        samples = np.genfromtxt(filename)
        if len(samples) == 0: # It's possible there's an empty file.
            samples = sampling_function(*model_params)
            save_data(filename, samples)
    except:
        # If there are no samples, we first generate a few.
        samples = sampling_function(*model_params)
        save_data(filename, samples)
    samples.sort()
    # Create the (r,s) pairs and check if the tolerance is met.
    # If the batch_size is too small this can cause a problem.
    pairs_list = [(0,0)]*len(required_quantiles)
    for i,p in enumerate(required_quantiles):
        pairs_list[i] = find_rs(p, samples.size, confidence)
    attempts = 1
    while not meets_tolerances(samples, pairs_list, abs_tolerance,rel_tolerance):
        print(f'\nContinuing the simulation with parameters {model_params}.')
        # If we've not met the tolerances then we generate more samples,
        # then test again.
        print(f'\nGenerating new samples! This is attempt number {attempts}.')
        attempts += 1
        new_samples = sampling_function(*model_params)
        save_data(filename, new_samples)
        new_samples.sort()
        samples = insert_new_elements(samples, new_samples)
        for i,p in enumerate(required_quantiles):
            pairs_list[i] = find_rs(p,samples.size,confidence)
    quantiles = np.empty(shape=(len(required_quantiles)))
    for i, p in enumerate(required_quantiles):
        quantiles[i] = samples[int(p*samples.size)-1]
    return quantiles

def ten_percentiles(abs_tolerance,rel_tolerance,confidence,filename,sampling_function,*model_params):
    """
    Computes the 10th, 20th, ..., 90th percentiles
    to the given tolerance and confidence.
    """
    desired_quantiles = np.linspace(start=0.1,stop=0.9,num=9)
    quantiles = compute_quantiles(desired_quantiles, abs_tolerance,rel_tolerance,confidence,filename,sampling_function,*model_params)
    return quantiles

