# CovXY

 Code which generated the diagrams in the paper "Largest nearest-neighbour link and connectivity threshold in a polytopal random sample" by Mathew D. Penrose, Xiaochuan Yang and Frankie Higgs. 

Code for the simulations in the paper "Covering one point process with another" by Frankie Higgs, Mathew D. Penrose and Xiaochuan Yang.

We place $n$ points $X_1, \dots, X_n$ in a region $A \subseteq \mathbb{R}^d$, which we think of as transmitters,
and $m$ points $Y_1, \dots, Y_m$ in a region $B \subseteq A$, which we think of as receivers.

We use each $X_i$ as the centre of a ball of radius $r$.
The two-sample $k$-coverage threshold $R_{n,m,k}$ is defined as the smallest $r$ so that each receiver $Y_j$ is within distance $r$ of at least $k$ transmitters.

In the paper we prove (for explicitly known constants $c_1$ and $c_2$) that if $m/n \to \tau$ for a $\tau > 0$ as $n \to \infty$, then $(\theta_d / |A|) n R_{n,m,k}^d - c_1 \log n - c_2 \log\log n$ (where $\theta_d$ is the volume of the $d$-dimensional unit ball, and $|A|$ is the volume of $A$) converges in distribution to a random variable, whose distribution we give.

This code generates samples of the convergent quantity $(\theta_d / |A|) n R_{n,m,k}^d - c_1 \log n - c_2 \log\log n$, and plots the empirical distribution on the same axes as the limiting distribution.
