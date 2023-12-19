import ball
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
We've generated all the data elsewhere.
The functions in this file load the data and plot diagrams.
They take a lot of arguments to manually tweak the diagrams.

In the ball or torus, the final arguments should be: tau, d, k, n
For BinsideA and the square, the final arguments are tau, k, n
(since the square is always 2-dimensional and the limit in BinsideA
doesn't depend on d).
"""

def plot_diagram(data_location,diagram_filename,xlow,xhigh,title,thm_number,lim_fn,corr_lim_fn,*model_params):
    """
    This function can produce almost all the diagrams.
    There are two other functions, one for the diagram with all the medians aligned,
    and one for the diagram with e^{-tau gamma_n}.
    
    Be careful: the limit function in ball.py and torus.py has parameters tau, d, k
    while the limit function in the other two files only has 2 parameters tau, k.
    model_params should be either (tau,d,k,n) or (tau,k,n) accordingly.
    """
    fig, ax = plt.subplots()
    samples = np.genfromtxt(data_location)
    samples.sort()
    ax.plot(samples, (np.arange(samples.size)+1)/samples.size, 'b', linewidth=2, label="Empirical distribution")
    my_range = np.arange(xlow, xhigh, 0.01)
    p_lim = lim_fn(my_range, *model_params[:-1])
    p_cor = corr_lim_fn(my_range, *model_params)
    ax.plot(my_range,p_lim,'k--',linewidth=2,label="Limiting cdf")
    ax.plot(my_range,p_cor,'r',linestyle='dotted',linewidth=3,label=f'Corrected cdf from Theorem {thm_number}')
    ax.set_ylim(0,1)
    ax.set_xlim(xlow,xhigh)
    ax.legend(loc='lower right')
    ax.set_title(title)
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    fig.set_size_inches(4.8, 6.4) # This gave us an ok size on the page.
    fig.savefig(diagram_filename)
    plt.close()

def limit_diagram(data_location,diagram_filename,xlow,xhigh,title,thm_number,lim_fn,*model_params):
    """
    Plots the same thing as plot_diagram, but without the corrected limit.
    model_params should be (tau,d,k) (for ball.py and torus.py),
    or (tau,k) (for square.py and BinsideA.py).
    """
    fig, ax = plt.subplots()
    samples = np.genfromtxt(data_location)
    samples.sort()
    ax.plot(samples, (np.arange(samples.size)+1)/samples.size, 'b', linewidth=2, label="Empirical distribution")
    my_range = np.arange(xlow, xhigh, 0.01)
    p_lim = lim_fn(my_range, *model_params)
    ax.plot(my_range,p_lim,'k--',linewidth=2,label=f'Limiting cdf from Theorem {thm_number}')
    ax.set_ylim(0,1)
    ax.set_xlim(xlow,xhigh)
    ax.legend(loc='lower right')
    ax.set_title(title)
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    fig.set_size_inches(4.8, 6.4) # This gave us an ok size on the page.
    fig.savefig(diagram_filename)
    plt.close()

def gamma_diagram(data_location,diagram_filename,xlow,xhigh,title,thm_number,*model_params):
    """
    Plots the diagram with exp(-tau * gamma_n) on it
    (i.e. the green dotted curve in Figure 3 of the paper).
    Note that computing gamma is quite slow, expect to wait
    at least 10 seconds.
    
    model_params should be (tau,d,k,n) as for plot_diagram.
    We don't specify the functions because everything is in ball.py
    """
    fig, ax = plt.subplots()
    samples = np.genfromtxt(data_location)
    samples.sort()
    ax.plot(samples, (np.arange(samples.size)+1)/samples.size, 'b', linewidth=2, label="Empirical distribution")
    my_range = np.arange(xlow, xhigh, 0.01)
    p_lim = ball.limit(my_range, *model_params[:-1])
    p_cor = ball.corrected_limit(my_range, *model_params)
    p_gam = np.exp(-model_params[0]*np.vectorize(ball.compute_gamma)(my_range, *model_params[1:]))
    ax.plot(my_range,p_lim,'k--',linewidth=2,label="Limiting cdf")
    ax.plot(my_range,p_cor,'r',linestyle='dotted',linewidth=3,label=f'Corrected cdf from Theorem {thm_number}')
    ax.plot(my_range,p_gam,'g',linestyle='dotted',linewidth=2,label='$e^{-\\tau_n \\gamma_n}$ from Lemma 4.1')
    ax.set_ylim(0,1)
    ax.set_xlim(xlow,xhigh)
    ax.legend(loc='lower right')
    ax.set_title(title)
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    fig.set_size_inches(4.8, 6.4) # This gave us an ok size on the page.
    fig.savefig(diagram_filename)
    plt.close()

def median0_diagram(data_location,diagram_filename,xlow,xhigh,title,thm_number,*model_params):
    """
    For the ball, plots the same data as plot_diagram
    and subtracts the sample median from the data,
    and the distribution median from the two limits,
    so every curve passes through (0, 0.5).
    
    model_params should be (tau,d,k,n)
    """
    fig, ax = plt.subplots()
    samples = np.genfromtxt(data_location)
    samples.sort()
    samples -= np.median(samples)
    ax.plot(samples, (np.arange(samples.size)+1)/samples.size, 'b', linewidth=2, label="Empirical distribution with median shifted to 0")
    my_range = np.arange(xlow, xhigh, 0.01)
    p_lim = ball.limit(my_range, *model_params[:-1],True)
    p_cor = ball.corrected_limit(my_range, *model_params,True)
    ax.plot(my_range,p_lim,'k--',linewidth=2,label="Limiting cdf with median shifted to 0")
    ax.plot(my_range,p_cor,'r',linestyle='dotted',linewidth=3,label=f'Corrected cdf from Thm {thm_number} with median shifted to 0')
    ax.set_ylim(0,1)
    ax.set_xlim(xlow,xhigh)
    ax.legend(loc='lower right')
    ax.set_title(title)
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    fig.set_size_inches(4.8, 6.4) # This gave us an ok size on the page.
    fig.savefig(diagram_filename)
    plt.close()


