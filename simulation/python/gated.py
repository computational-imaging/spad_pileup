
# coding: utf-8

# In[46]:

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


# In[47]:

data = scipy.io.loadmat('Data/cornell_indirect.mat')
data_mat = data['cornell_indirect']
# data_mat = data_mat[:150,130:150,130:150].astype('float64')
data_mat = data_mat[:150,:,:].astype('float64')
print data_mat.max()
print "shape = ", data_mat.shape
T, dim1, dim2 = data_mat.shape

# In[55]:

# Normalize all intensity profiles.
for idx1 in range(dim1):
    for idx2 in range(dim2):
        sum_val = data_mat[:,idx1,idx2].sum()
        if sum_val != 0:
            data_mat[:,idx1,idx2] /= sum_val


# In[56]:


# Simulate gated.
jitter_on = True
# vals = np.arange(-(T-1),(T-1)+1)
# g = mlab.normpdf(vals,0, sigma_spad/100e-12);
# probs = g/g.sum()
jitter_probs = [0.1, 0.6, 0.2, 0.1]
# jitter_probs = [0.1, 0.6, 0.3]
jitter_vals = [-1, 0, 1]
jitter_distr = rv_discrete(a=-1, values=(jitter_vals, jitter_probs))

# G = np.cumsum(g)
NUM_TRIALS = 5000 # Number of shots

GATE_SIZE = 10

import numba
@numba.jit(nopython=True)
def get_buckets(NUM_TRIALS, data_mat, jitter_on, jitter_vals, jitter_probs):
    T, dim1, dim2 = data_mat.shape
    buckets = np.zeros((T, dim1, dim2))
    for trial in range(NUM_TRIALS):
        start_gate = GATE_SIZE*trial % T
        end_gate = start_gate + GATE_SIZE
#         if trial % 50 == 0:
#             print(trial)
        for idx1 in range(dim1):
            for idx2 in range(dim2):
                r = data_mat[:,idx1,idx2]
                lamb = QE*(I*r + amb) + DC
                min_so_far = np.inf # min of T means not detected.
                for t in range(start_gate,end_gate):
                    nt = np.random.poisson(lamb[t])
                    if nt > 0:
                        if jitter_on:
                            jit = np.random.choice(jitter_vals, nt, jitter_probs)
                            hit = t + jit.min()
                            if hit < min_so_far:
                                if start_gate <= hit < end_gate:
                                    min_so_far = hit
                                # Project firing times onto [0,T-1].
                                elif hit < start_gate:
                                    min_so_far = start_gate
                                else:
                                    min_so_far = end_gate - 1
                        else:
                            hit = t
                            min_so_far = min(min_so_far, hit)
                if min_so_far < np.inf:
                    buckets[int(min_so_far), idx1, idx2] += 1

    return buckets
if True:
    buckets = get_buckets(NUM_TRIALS, data_mat, jitter_on, np.array(jitter_vals), jitter_probs)
    # np.save('gated_hist', buckets)
else:
    buckets = np.load('gated_hist.npy')
# print hist
print "NUM TRIALS = ", NUM_TRIALS
true_r = data_mat

# Normalize buckets.
for idx1 in range(dim1):
    for idx2 in range(dim2):
        sum_val = buckets[:,idx1,idx2].sum()
        if sum_val != 0:
            buckets[:,idx1,idx2] /= sum_val
# np.save('experiments/Gated_cornell', buckets)

print "||gated - r_true||^2_2 = ", np.square(buckets - true_r).sum()

