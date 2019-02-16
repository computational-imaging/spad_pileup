import numpy as np
from scipy.stats import norm, poisson, rv_discrete
import proximal as pr
import numba


I = 10 # photons/laser pulse
QE = .30; #Quantum efficiency
DC = 1e-5;  # dark counts per exposure
amb = 1e-5; # ambient light

jitter_probs = [0.1, 0.6, 0.2, 0.1]
jitter_vals = [-1, 0, 1, 2]
jitter_distr = rv_discrete(a=-1, values=(jitter_vals, jitter_probs))

NUM_TRIALS = 5000 # Number of shots

SIZE = 2
LEN = 100
hist = np.load('hist.npy')[:LEN,:SIZE,:SIZE]
true_r = np.load('experiments/True_cornell.npy')[:LEN,:SIZE,:SIZE]
T, dim1, dim2 = hist.shape
print hist.shape
print hist.size

# Get Coates estimate.
@numba.jit(nopython=True)
def coates(T, N, h):
    r = np.zeros(T)
    r[0] = -np.log(1-h[0]/N)
    for k in range(1,T):
        tmp = N - np.sum(h[:k])
        r[k] = -np.log(1-h[k]/tmp)
    return r

r_coates = np.zeros(hist.shape)
for idx1 in range(dim1):
    for idx2 in range(dim2):
        lamb_c = coates(T, NUM_TRIALS, hist[:,idx1,idx2])
        r_coates[:,idx1,idx2] = np.maximum( ((lamb_c - DC)/QE - amb)/I, 0)

# Solve without jitter first.

# Make G1, G2 matrices from hist.
# G1 = norm.cdf(hist - 1 - times, loc=0, scale=jitter)
# G2 = norm.cdf(hist - times, loc=0, scale=jitter)
times = np.arange(T)
if False:
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

# lamb = QE*(I*r + amb) + DC
offset = QE*amb + DC

@numba.jit(nopython=True)
def make_c(G1, hist):
    c = np.zeros(hist.shape)
    for idx1 in range(dim1):
        for idx2 in range(dim2):
            buckets = hist[:,idx1,idx2]
            c[:,idx1,idx2] = np.dot(buckets, G1) + (NUM_TRIALS - buckets.sum())*np.ones(T)
    return c
c = make_c(G1, hist)

r = pr.Variable(hist.shape)
def grad(x):
    val = np.exp(-x-offset).ravel()
    return -hist.ravel()*val/(1-val)

def fn_val(x):
    return -np.dot(hist.ravel(), np.log(1-np.exp(-x-offset)).ravel())

SCALE = 1e4
log_exp = pr.diff_fn(r, fn_val, grad,
                     bounds=r.size*[(0,None)], factr=10,
                     beta=QE*I, c=c*QE*I*SCALE**-.5, alpha=SCALE**-.5)
prob = pr.Problem([log_exp, pr.nonneg(r), pr.norm1(pr.grad(r), alpha=SCALE**.5)],
                  solver='admm', halide=True, lin_solver='cg')
print "Starting to solve"
prob.solve(verbose=True, max_iters=10, eps_abs=1e-5, eps_rel=1e-5, rho=100,
    x0=r_coates)
r_MAP = np.maximum(r.value, 0)


for idx1 in range(dim1):
    for idx2 in range(dim2):
        sum_val = true_r[:,idx1,idx2].sum()
        if sum_val != 0:
            true_r[:,idx1,idx2] /= sum_val
error_prior = np.square(true_r - r_MAP).sum()
error_coates = np.square(true_r - r_coates).sum()
print "prior coates error =", np.square(r_coates - r_MAP).sum()
print "error prior = ", error_prior
print "error coates =", error_coates
print "MSE ratio =", error_prior/error_coates
