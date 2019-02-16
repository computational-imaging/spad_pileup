# Test on transient image.

from __future__ import division
from scipy.stats import norm, poisson, rv_discrete
import scipy.io
import numpy as np
import matplotlib.mlab as mlab

np.random.seed(0)
# Constants.
I = 10 # photons/laser pulse
QE = .30; #Quantum efficiency
DC = 1e-5;  # dark counts per exposure
amb = 1e-5; # ambient light

mu_laser = 0
wavelength = 455e-9 # nanometers
FWHM_laser = 128e-12 # 128 ps
sigma_laser = FWHM_laser/(2*np.sqrt(2*np.log(2)))
# FWHM = 2 sqrt(2ln(2))sigma
mu_spad = 0
FWHM_spad = 70e-12 # 70 ps
sigma_spad = FWHM_spad/(2*np.sqrt(2*np.log(2)))

data = scipy.io.loadmat('Data/bunny_indirect.mat')
data_mat = data['bunny_indirect']
T, dim1, dim2 = data_mat.shape
print data_mat[:,0,0]
r = r/np.sum(r);
R = np.cumsum(r);

#Create expected histogram probability with jitter
jitter_on = True
# vals = np.arange(-(T-1),(T-1)+1)
# g = mlab.normpdf(vals,0, sigma_spad/100e-12);
# probs = g/g.sum()
probs = [0.1, 0.6, 0.3]
vals = [-1, 0, 1]
jitter_distr = rv_discrete(a=-1, values=(vals, probs))

# G = np.cumsum(g)

lamb = QE*(I*r + amb) + DC

NUM_TRIALS = 5000 # Number of shots
hist = np.zeros((NUM_TRIALS, 1))
for trial in range(NUM_TRIALS):
    counts = np.random.poisson(lamb)
    min_so_far = np.inf # min of T means not detected.
    for t, nt in enumerate(counts):
        if nt > 0:
            if jitter_on:
                jit = jitter_distr.rvs(size=nt)
                hits = jit + t
            else:
                hits = t
            if np.min(hits) < min_so_far:
                if 0 <= np.min(hits) < T:
                    min_so_far = np.min(hits)
                # Project firing times onto [0,T-1].
                elif np.min(hits) < 0:
                    min_so_far = 0
                else:
                    min_so_far = T - 1

    hist[trial] = min_so_far

NUM_TRIALS = len(hist)
print "NUM TRIALS = ", NUM_TRIALS
true_r = r

# Get histogram:
buckets = np.zeros(T)
for t in range(T):
    buckets[t] = np.sum(hist == t)
total_misses = np.sum(hist == np.inf)
print "misses = ", total_misses
print buckets

