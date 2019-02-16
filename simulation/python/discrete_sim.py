# This code creates a histogram of photon detections, and then creates the
# Coates and MAP solutions.

from __future__ import division
from scipy.stats import norm, poisson, rv_discrete
import numpy as np
import matplotlib.mlab as mlab

np.random.seed(0)
# Constants.
I = 1 # photons/laser pulse
QE = .30; #Quantum efficiency
DC = 1e-5;  # dark counts per exposure
amb = .1; # ambient light

T = 100;
times = np.arange(0,T)
tmp = np.linspace(0,1,T);

# Choose the pulse here.
r = np.exp(-(tmp-.5)**2/.05);
# r = np.abs(np.random.randn(T))
# r = np.sin(tmp)

r = r/np.sum(r);
R = np.cumsum(r);

#Create expected histogram probability with jitter
jitter_on = False
# Choose the jitter distribution here.
jitter_probs = [0.1, 0.6, 0.2, 0.1]
jitter_vals = [-1, 0, 1, 2]
jitter_distr = rv_discrete(a=-1, values=(jitter_vals, jitter_probs))

# G = np.cumsum(g)

lamb = QE*(I*r + amb) + DC

NUM_TRIALS = int(5e4) # Number of shots
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
total_misses = NUM_TRIALS - buckets.sum()
print "misses = ", total_misses

# Simulation with gates.
GATE_SIZE = 10 # Number of bins active at once.
gated_hist = np.zeros(T)
for trial in range(NUM_TRIALS):
    start_gate = GATE_SIZE*trial % T
    end_gate = start_gate + GATE_SIZE
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
        gated_hist[int(min_so_far)] += 1

def G_func(T, t, times):
    # Counts t < 0 as t = 0 and t > T-1 as t = T-1.
    if t == -1:
        return np.zeros(T)
    elif t == T - 1:
        return np.ones(T)
    else:
        return jitter_distr.cdf(t - times)

# Make G1, G2 matrices from hist.
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

# Get MAP estimate.
import cvxpy as cvx

r = cvx.Variable(T)
lamb = QE*(I*r + amb) + DC
t = cvx.Variable(NUM_TRIALS)

# Gordon: choose the regularization/prior here.
# It's gamma*reg
gamma = 5e5
reg = cvx.sum_squares(r[1:] - r[:-1])

expr = cvx.exp(-G1*lamb) - cvx.exp(-G2*lamb)
miss = -cvx.sum_entries(lamb)
ll_orig = buckets.T*cvx.log(expr) + miss*total_misses

tmp = cvx.log(1 - cvx.exp((G1-G2)*lamb)) - G1*lamb
ll = buckets.T*tmp + miss*total_misses

prob = cvx.Problem(cvx.Maximize(ll/np.sqrt(gamma) - np.sqrt(gamma)*reg)/NUM_TRIALS, [r >= 0])
# prob = cvx.Problem(cvx.Maximize(ll)/NUM_TRIALS, [r >= 0])
prob.solve(solver=cvx.ECOS, verbose=False)
print "status = ", prob.status
r_star = r.value.A.ravel()

print "||r - r_true||_2/||r_true||_2 = ", np.linalg.norm(r_star - true_r)/np.linalg.norm(true_r)

# print ll_orig.value
print "ll value at r =", ll.value

# print "true r"
# print true_r
r.value = true_r
print "ll value at true r =", ll.value

def coates(T, N, h):
    r = np.zeros(T)
    r[0] = -np.log(1-h[0]/N)
    for k in range(1,T):
        tmp = N - np.sum(h[:k])
        r[k] = -np.log(1-h[k]/tmp)
    return r

lamb_c = coates(T, NUM_TRIALS, buckets)
r_c = ((lamb_c - DC)/QE - amb)/I
print "||r_c - r_true||_2/||r_true||_2 = ", np.linalg.norm(r_c - true_r)/np.linalg.norm(true_r)

import matplotlib.pyplot as plt
plt.plot(times,true_r/true_r.sum(), linestyle='--')
plt.plot(times,gated_hist/gated_hist.sum())
plt.plot(times,r_star/r_star.sum(), linestyle=':')
plt.plot(times,r_c/r_c.sum())
plt.plot(times, buckets/buckets.sum())
plt.legend(['True','Gated','MAP','Coates','Raw histogram']);
plt.xlabel('time');
plt.ylabel('r_t');
plt.title('Normalized Intensities')
# plt.show()

plt.savefig('outdoors.png')
