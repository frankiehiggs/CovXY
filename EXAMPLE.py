"""
This is an example of how to use the code to generate samples
and plot the appropriate diagrams.
"""

"""
To generate a large number of samples,
we only need to call the function ten_percentiles
from quantiles.py
Also import generate_R_samples from whichever of the other files
you like, ball.py, BinsideA.py, square.py or torus.py
"""
from quantiles import ten_percentiles
from torus import generate_R_samples

"""
We will save the data in a csv file.
Choose a set of parameters for the simulation,
how closely we want to estimate the quantiles,
and the batch size (i.e. how often we will save the samples to the file)
"""
filename = 'exampledata.csv'
n,m,d,k = 1000,1000,2,1
width,confidence = 0.1,0.95
batch_size = 2000

qs = ten_percentiles(width,width,confidence,filename,generate_R_samples,n,m,d,k,batch_size)
for i,q in enumerate(qs):
    print(f'The {10*(i+1)}th percentile is {q:.3f}.')

"""
After ten_percentiles has finished, it will return the percentiles,
and our csv now contains a lot of samples.

For the diagrams in the paper, we took width=0.1 and confidence=0.95,
apart from the diagram generated using BinsideA.py (the top-left
of Figure 1), which I left running overnight and has width around 0.06.

Next, we plot the diagrams.
"""

