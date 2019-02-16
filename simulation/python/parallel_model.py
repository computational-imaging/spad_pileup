
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
data_mat = data_mat[:150,:,:].astype('float64')
# data_mat = data_mat[:150,:,:].astype('float64')
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


# Simulate histograms.
jitter_on = False
# vals = np.arange(-(T-1),(T-1)+1)
# g = mlab.normpdf(vals,0, sigma_spad/100e-12);
# probs = g/g.sum()
jitter_probs = [0.05, 0.8, 0.15]
# jitter_probs = [0.1, 0.6, 0.3]
jitter_vals = [-1, 0, 1]
jitter_distr = rv_discrete(a=-1, values=(jitter_vals, jitter_probs))

# G = np.cumsum(g)
NUM_TRIALS = 5000 # Number of shots

import numba
@numba.jit(nopython=True)
def get_buckets(NUM_TRIALS, data_mat, jitter_on, jitter_vals, jitter_probs):
    T, dim1, dim2 = data_mat.shape
    buckets = np.zeros((T, dim1, dim2))
    for trial in range(NUM_TRIALS):
#         if trial % 50 == 0:
#             print(trial)
        for idx1 in range(dim1):
            for idx2 in range(dim2):
                r = data_mat[:,idx1,idx2]
                lamb = QE*(I*r + amb) + DC
                min_so_far = np.inf # min of T means not detected.
                for t in range(T):
                    nt = np.random.poisson(lamb[t])
                    if nt > 0:
                        if jitter_on:
                            jit = np.random.choice(jitter_vals, nt, jitter_probs)
                            hit = t + jit.min()
                            if hit < min_so_far:
                                if 0 <= hit < T:
                                    min_so_far = hit
                                # Project firing times onto [0,T-1].
                                elif hit < 0:
                                    min_so_far = 0
                                else:
                                    min_so_far = T - 1
                        else:
                            hit = t
                            min_so_far = min(min_so_far, hit)
                if min_so_far < np.inf:
                    buckets[int(min_so_far), idx1, idx2] += 1

    return buckets

if True:
    buckets = get_buckets(NUM_TRIALS, data_mat, jitter_on, np.array(jitter_vals), jitter_probs)
else:
    buckets = np.load('no_jitter_hist.npy')
# print hist
print "NUM TRIALS = ", NUM_TRIALS
true_r = data_mat



# In[ ]:

# Recover every pixel.
times = np.arange(0,T)
def G_func(T, t, times):
    # Counts t < 0 as t = 0 and t > T-1 as t = T-1.
    if t == -1:
        return np.zeros(T)
    elif t == T - 1:
        return np.ones(T)
    else:
        return jitter_distr.cdf(t - times)

# Make G1, G2 matrices from hist.
# G1 = norm.cdf(hist - 1 - times, loc=0, scale=jitter)
# G2 = norm.cdf(hist - times, loc=0, scale=jitter)
if jitter_on:
    G1 = np.zeros((T, T))
    G2 = np.zeros((T, T))
    for t in range(T):
        G1[t,:] = G_func(T, t-1, times)
        G2[t,:] = G_func(T, t, times)
else:
    # No jitter.
    # Position T indicates no detection.
    G1 = np.zeros((T, T))
    G2 = np.zeros((T, T))
    for t in range(T):
        G1[t,:] = times <= t - 1
        G2[t,:] = times <= t

@numba.jit
def coates(T, N, h):
    r = np.zeros(T)
    r[0] = -np.log(1-h[0]/N)
    for k in range(1,T):
        tmp = N - np.sum(h[:k])
        r[k] = -np.log(1-h[k]/tmp)
    return r

print "+"*25
# Get MAP estimate.
import cvxpy as cvx
# import epopt as ep


error_prior = 0
error_coates = 0
r_coates_full = np.zeros((T,dim1, dim2))
r_recov_full = np.zeros((T,dim1, dim2))

def get_estimate(idxs):
    idx1, idx2 = idxs
    r_slc = true_r[:,idx1,idx2]
    r = cvx.Variable(T)
    lamb = QE*(I*r + amb) + DC
    gamma = cvx.Parameter(sign="positive")
    reg = cvx.sum_squares(r[1:] - r[:-1]) #+ r[-1]**2 + r[0]**2
    # reg += 0.01*cvx.sum_entries(r)
    # reg = cvx.tv(r)
    # diff = r[2:] - 2*r[1:-1] + r[:-2]
    # reg = cvx.norm(diff, 1)

    total_misses = NUM_TRIALS - buckets[:,idx1,idx2].sum()
    miss = -cvx.sum_entries(lamb)
    tmp = cvx.log(1 - cvx.exp((G1-G2)*lamb)) - G1*lamb
    ll = buckets[:,idx1,idx2].T*tmp + miss*total_misses

    gamma = 10
    prob = cvx.Problem(cvx.Minimize(-ll/(NUM_TRIALS*np.sqrt(gamma)) + np.sqrt(gamma)*reg), [lamb >= 0])
    # prob = cvx.Problem(cvx.Maximize(ll)/NUM_TRIALS, [r >= 0])
    try:
        prob.solve()
        # print prob.status
    except cvx.SolverError:
        prob.solve(solver=cvx.SCS, eps=1e-4)

    # error_prior += cvx.sum_squares(cvx.pos(r) - r_slc)
#         print "||r - r_true||_2^2 = ", np.linalg.norm(r_star - r_slc)**2#/np.linalg.norm(r_slc)

    # print ll_orig.value
#         print "ll value at r =", ll.value

    lamb_c = coates(T, NUM_TRIALS, buckets[:,idx1,idx2])
    r_c = np.maximum( ((lamb_c - DC)/QE - amb)/I, 0)
#         print "||r_c - r_true||_2^2 = ", np.linalg.norm(r_c - r_slc)**2#/np.linalg.norm(r_slc)
    # error_coates += np.linalg.norm(r_c - r_slc)**2
    # print idx1, idx2
    return (r_c, r.value.A.ravel())

import multiprocessing
import itertools
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
idxs = list( itertools.product(range(dim1), range(dim2)) )
solve_results = pool.map(get_estimate, idxs)

for (r_c,r_star), (idx1, idx2) in zip(solve_results, idxs):
    r_coates_full[:,idx1,idx2] = np.maximum(r_c, 0)
    r_recov_full[:,idx1,idx2] = np.maximum(r_star, 0)
    r_slc = data_mat[:,idx1,idx2]
    error_prior += np.linalg.norm(r_star - r_slc)**2
    error_coates += np.linalg.norm(r_c - r_slc)**2

# prob = 0
# r_vars = {}
# for idx1 in range(dim1):
#     for idx2 in range(dim2):
#         r = cvx.Variable(T)
#         r_vars[(idx1,idx2)] = r

# for idx1 in range(dim1):
#     for idx2 in range(dim2):
#         r_slc = true_r[:,idx1,idx2]
#         r = r_vars[(idx1, idx2)]
#         lamb = QE*(I*r + amb) + DC
#         gamma = cvx.Parameter(sign="positive")
# #         reg = cvx.sum_squares(r[1:] - r[:-1])
#         reg = cvx.norm(r,1)*.05
#         # reg = cvx.tv(r)
#         # reg = 0
#         if idx1+1 < dim1:
#             reg += cvx.norm(r_vars[(idx1+1,idx2)] - r_vars[(idx1,idx2)], 1)
#         if idx2+1 < dim2:
#             reg += cvx.norm(r_vars[(idx1,idx2+1)] - r_vars[(idx1,idx2)], 1)

#         total_misses = NUM_TRIALS - buckets[:,idx1,idx2].sum()
#         miss = -cvx.sum_entries(lamb)
#         tmp = cvx.log(1 - cvx.exp((G1-G2)*lamb)) - G1*lamb
#         ll = buckets[:,idx1,idx2].T*tmp + miss*total_misses

#         gamma = .05
#         prob += cvx.Problem(cvx.Maximize(ll/(NUM_TRIALS*np.sqrt(gamma)) - np.sqrt(gamma)*reg), [r >= 0])
#         # prob += cvx.Problem(cvx.Maximize(ll)/NUM_TRIALS, [r >= 0])
#         error_prior += cvx.sum_squares(cvx.pos(r) - r_slc)
# #         print "||r - r_true||_2^2 = ", np.linalg.norm(r_star - r_slc)**2#/np.linalg.norm(r_slc)

#         # print ll_orig.value
# #         print "ll value at r =", ll.value

#         lamb_c = coates(T, NUM_TRIALS, buckets[:,idx1,idx2])
#         r_c = np.maximum( ((lamb_c - DC)/QE - amb)/I, 0)
#         r_coates_full[:,idx1,idx2] = r_c
# #         print "||r_c - r_true||_2^2 = ", np.linalg.norm(r_c - r_slc)**2#/np.linalg.norm(r_slc)
#         error_coates += np.linalg.norm(r_c - r_slc)**2

# prob.solve(verbose=True, solver=cvx.ECOS, max_iters=200)

# error_prior = error_prior.value
# for idx1 in range(dim1):
#     for idx2 in range(dim2):
#         r_star = r_vars[(idx1,idx2)].value.A.ravel()
#         r_recov_full[:,idx1,idx2] = np.maximum(r_star, 0)

print "||r_star - r_true||_2^2 = ", error_prior/np.sqrt(T*dim1*dim2)
print "||r_c - r_true||_2^2 = ", error_coates/np.sqrt(T*dim1*dim2)
print "MSE ratio = ", error_prior/error_coates

imgs = {}
imgs['True'] = data_mat
imgs['Coates'] = r_coates_full
imgs['MAP'] = r_recov_full

from matplotlib import pyplot as plt
for k, frame in enumerate([30, 60, 100]):
    for i, (name, image_arr) in enumerate(imgs.items()):
        t = frame
        if name == 'True':
            t -= 1
        fig = plt.subplot(5, 3, i+1 + k*3)
        plt.imshow(image_arr[t,:,:], cmap='Greys_r')
        if k == 0:
            plt.title(name)
        if i == 0:
            plt.ylabel('Frame %d' % frame)
        fig.axes.get_xaxis().set_ticks([])
        fig.axes.get_yaxis().set_ticks([])
# Plot individual pixels.
for k, (idx1,idx2,label) in enumerate([(10,10,'Large box')]):
    for i, (name, image_arr) in enumerate(imgs.items()):
        fig = plt.subplot(5, 3, i+1 + 9+k*3)
        plt.plot(image_arr[:,idx1,idx2])
        if i == 0:
            plt.ylabel(label)
        fig.axes.get_xaxis().set_ticks([])
        fig.axes.get_yaxis().set_ticks([])
        # fig.axes.set_aspect('equal', 'datalim')

plt.show()

# np.save('coates_cornell', r_coates_full)
# np.save('recov_cornell', r_recov_full)
