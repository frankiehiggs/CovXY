"""
Run this script to generates all the diagrams used in the paper,
and all the data needed for these diagrams. It takes a very long time
to run, but saves the generated data regularly, so you can safely
stop it and resume later.
"""
import ball, BinsideA, square, torus
from quantiles import ten_percentiles
import diagrams
import os

if __name__=='__main__':
    # Parameters
    W = 0.1
    C = 0.95
    EXT = '.pdf' # We can change the extension to .eps if we like
    
    if not os.path.exists('data/'):
        os.makedirs('data/')
    if not os.path.exists('diagrams/'):
        os.makedirs('diagrams/')
    # Diagram 1:
    # BinsideA, n=10**6, m=n, d=2, k=1,
    # without the corrected limit.
    # This is by far the slowest of the simulations
    # since n is 100 times larger than the others.
    # This is because for smaller n we seemed to still see
    # a strong boundary effect.
    fileprefix = 'b-inside-a-n1e6-tau1-d2-k1'
    datafile = 'data/'+fileprefix+'.csv'
    diagfile = 'diagrams/'+fileprefix+EXT
    n,tau,d,k = 1000000,1,2,1
    m = tau*n
    batch_size = 25
    xlow,xhigh = -2, 4
    title = '$n=10^6$, $\\tau_n = 1$, $d=2$ and $k=1$ for $B(o,0.9) \subseteq B(o,1)$'
    thm_num = '2.1'
    ten_percentiles(W,W,C,datafile,BinsideA.generate_R_samples,n,m,d,k,batch_size)
    diagrams.limit_diagram(datafile,diagfile,xlow,xhigh,title,thm_num,BinsideA.limit,tau,k)
    
    # Diagram 2:
    # torus, n=10**4, m=n, d=2, k=3
    fileprefix = 'torus-n1e4-tau1-d2-k3'
    datafile = 'data/'+fileprefix+'.csv'
    diagfile = 'diagrams/'+fileprefix+EXT
    n,tau,d,k = 10000,1,2,3
    m = tau*n
    batch_size = 200
    xlow,xhigh = -2, 4
    title = '$n=10^4$, $\\tau_n = 1$, $d=2$ and $k=3$ in a torus'
    thm_num = '2.1'
    ten_percentiles(W,W,C,datafile,torus.generate_R_samples,n,m,d,k,batch_size)
    diagrams.plot_diagram(datafile,diagfile,xlow,xhigh,title,thm_num,torus.limit,torus.corrected_limit,tau,d,k,n)
    
    # Diagram 3:
    # ball, n=10**4, m=n, d=2, k=1
    fileprefix = 'ball-n1e4-tau1-d2-k1'
    datafile = 'data/'+fileprefix+'.csv'
    diagfile = 'diagrams/'+fileprefix+EXT
    n,tau,d,k = 10000,1,2,1
    m = tau*n
    batch_size = 5000
    xlow,xhigh = -2, 6
    title = '$n=10^4$, $\\tau_n = 1$, $d=2$ and $k=1$ in the unit radius disc'
    thm_num = '2.2'
    ten_percentiles(W,W,C,datafile,ball.generate_R_samples,n,m,d,k,batch_size)
    diagrams.plot_diagram(datafile,diagfile,xlow,xhigh,title,thm_num,ball.limit,ball.corrected_limit,tau,d,k,n)
    
    # Diagram 4:
    # same as 3, but with medians = 0
    diagfile = 'diagrams/'+fileprefix+'-medians-subtracted'+EXT
    diagrams.median0_diagram(datafile,diagfile,xlow,xhigh,title,thm_num,tau,d,k,n)
    
    # Diagram 5:
    # square, n=10**4, m=n, d=2, k=1
    fileprefix = 'square-n1e4-tau1-d2-k1'
    datafile = 'data/'+fileprefix+'.csv'
    diagfile = 'diagrams/'+fileprefix+EXT
    n,tau,d,k = 10000,1,2,1
    m = tau*n
    batch_size = 6000
    xlow,xhigh = -2, 6
    title = '$n=10^4$, $\\tau_n = 1$, $d=2$ and $k=1$ in the square $[0,1]^2$'
    thm_num = '2.2'
    ten_percentiles(W,W,C,datafile,square.generate_R_samples,n,m,k,batch_size)
    diagrams.plot_diagram(datafile,diagfile,xlow,xhigh,title,thm_num,square.limit,square.corrected_limit,tau,k,n)
    
    # Diagram 6:
    # ball, n=10**4, m=n, d=2, k=2
    fileprefix = 'ball-n1e4-tau1-d2-k2'
    datafile = 'data/'+fileprefix+'.csv'
    diagfile = 'diagrams/'+fileprefix+EXT
    n,tau,d,k = 10000,1,2,2
    m = tau*n
    batch_size = 5001
    xlow,xhigh = -2, 8
    title = '$n=10^4$, $\\tau_n = 1$, $d=2$ and $k=2$ in the unit radius disc'
    thm_num = '2.3'
    ten_percentiles(W,W,C,datafile,ball.generate_R_samples,n,m,d,k,batch_size)
    diagrams.plot_diagram(datafile,diagfile,xlow,xhigh,title,thm_num,ball.limit,ball.corrected_limit,tau,d,k,n)
    
    # Diagram 7:
    # ball, n=10**4, m=n, d=3, k=1,
    # with gamma
    fileprefix = 'ball-n1e4-tau1-d3-k1'
    datafile = 'data/'+fileprefix+'.csv'
    diagfile = 'diagrams/'+fileprefix+EXT
    n,tau,d,k = 10000,1,3,1
    m = tau*n
    batch_size = 4000
    xlow,xhigh = -1, 8
    title = '$n=10^4$, $\\tau_n = 1$, $d=3$ and $k=1$ in the unit radius ball'
    thm_num = '2.3'
    ten_percentiles(W,W,C,datafile,ball.generate_R_samples,n,m,d,k,batch_size)
    diagrams.gamma_diagram(datafile,diagfile,xlow,xhigh,title,thm_num,tau,d,k,n)
    
    # Diagram 8:
    # ball, n=10**4, m=100*n, d=3, k=1,
    # with gamma.
    fileprefix = 'ball-n1e4-tau100-d3-k1'
    datafile = 'data/'+fileprefix+'.csv'
    diagfile = 'diagrams/'+fileprefix+EXT
    n,tau,d,k = 10000,100,3,1
    m = tau*n
    batch_size = 50
    xlow,xhigh = 6, 17
    title = '$n=10^4$, $\\tau_n = 100$, $d=3$ and $k=1$ in the unit radius ball'
    thm_num = '2.3'
    ten_percentiles(W,W,C,datafile,ball.generate_R_samples,n,m,d,k,batch_size)
    diagrams.gamma_diagram(datafile,diagfile,xlow,xhigh,title,thm_num,tau,d,k,n)
