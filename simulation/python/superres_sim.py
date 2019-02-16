from __future__ import division
from scipy.stats import norm, poisson
import numpy as np

np.random.seed(0)

# Constants.
I = 50 # photons/laser pulse
QE = .30; #Quantum efficiency
DC = 0;  # dark counts per exposure
amb = 0; # ambient light
c = 3e8 # speed of light

d = 3.75e-3 # distance to target

mu_laser = 0
wavelength = 455e-9 # nanometers
FWHM_laser = 500e-12 # high noise
# FWHM_laser = 128e-12 # 128 ps, low noise
sigma_laser = FWHM_laser/(2*np.sqrt(2*np.log(2)))
# FWHM = 2 sqrt(2ln(2))sigma
mu_spad = 0
FWHM_spad = 500e-12 # for high noise
# FWHM_spad = 70e-12 # 70 ps, low noise
sigma_spad = FWHM_spad/(2*np.sqrt(2*np.log(2)))
# jitter = laser jitter + SPAD jitter
# = N(mu_laser + mu_spad, sigma_spad**2 + sigma_laser**2)

# CDF G(t) = 1 - (1 - F(t))^n
# G^{-1}(t) = F^{-1}(1 - (1 - G(t))^{1/n})
# Here F(t) = jitter CDF

# Model.
# photons generated (Poisson) -> iid jitter -> take min
# Poisson mean = I*QE + ambient + DC
reps = 1000; # Number of shots

mean = I*QE
counts = np.random.poisson(mean, size=reps)
# Get times via sample inverse CDF(unif [0,1])
unifs = np.random.uniform(size=reps)
times = norm.ppf( (1 - (1 - unifs)**(1./counts)), loc=mu_laser+mu_spad,
                  scale=np.sqrt(sigma_spad**2 + sigma_laser**2) )
# print "times = ", times
# Make histogram.
# 100 ps per bucket, 255 buckets
bucket_size = 100e-12
num_buckets = 100

buckets = np.arange(-50, 50)*bucket_size
rounded = np.floor(times/bucket_size).astype(int)
min_off = np.min(rounded)
max_off = np.max(rounded)
# print("max offset = %s, min offset = %s" % (max_off, min_off))
pos_rounded = np.concatenate( [rounded + num_buckets//2, range(num_buckets)])

hist = np.bincount(pos_rounded) - 1
# print "hist = ", hist

def F(times, offset):
    return norm.cdf(times, loc=mu_laser+mu_spad+offset,
        scale=np.sqrt(sigma_spad**2 + sigma_laser**2))

def get_pdf(offset, hist, buckets):
    # Get MAP estimate of initial time.
    # Pr(observations | t_0) = \Prod_{i=1}^N Pr(t_i in bucket I_i).
    # = Pr(t_i in bucket I_i) = \sum_{n=0}^\infty Pr(t_i in bucket I_i | n)Pr(n)
    # Pr(t_i in bucket I_i | n) = CDF(I_i max | n) - CDF(I_i min | n)
    bucket_probs = np.exp(-F(buckets, offset)*mean) - np.exp(-F(buckets + bucket_size, offset)*mean)
    prob = hist*np.log(np.maximum(bucket_probs, 1e-100))
    return prob.sum()

num_offsets = 10001
offsets = np.linspace(-bucket_size, bucket_size, num=num_offsets)
masses = np.zeros(num_offsets)
for i in range(num_offsets):
    masses[i] = get_pdf(offsets[i], hist, buckets)
print "spatial resolution in mm = ", c*bucket_size*1000
map_estimate = offsets[np.argmax(masses)]
print "MAP estimate in mm = ", c*map_estimate/2*1000

# Now fit a Gaussian.
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
SCALE = 1e9
x = (buckets + bucket_size/2)*SCALE
n = len(x)
hist_scaled = hist/hist.sum()
mean = sum(x*hist_scaled)/n                   #note this correction
sigma = sum(hist_scaled*(x-mean)**2)/n        #note this correction
def gauss(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
popt,pcov = curve_fit(gauss,x,hist_scaled,p0=[hist_scaled.max(),mean,sigma])
_, mean, var = popt
print mean, var
estimate = mean/SCALE
print "Gaussian fit in mm = ", c*estimate/2*1000
print 500*c*(buckets[np.argmax(hist)] + bucket_size/2)

# MAP estimate in mm =  0.081 for low noise
#Gaussian fit in mm =  -15.7950009204

# MAP estimate in mm =  0.324 for high noise
# Gaussian fit in mm =  -76.8749998707

# Plot histogram
import matplotlib.pyplot as plt
# the histogram of the data
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.hist(pos_rounded, bins=100)
plt.xlabel('Time')
plt.ylabel('Histogram count')
plt.title('Non-Gated SPAD')

plt.subplot(1,2,2)
plt.plot(offsets*c/2*1000, masses, label='Bayesian pdf')
# plt.plot(offsets, masses, label='Gaussian fit')
plt.axvline(x=estimate*c/2*1000, linestyle='-.', label='Gaussian fit')
plt.axvline(x=0, linestyle='--', label='Target', color='red')
plt.axvline(x=map_estimate*c/2*1000, linestyle=':', label='Bayesian fit')
plt.ylim([-2000,-1900])
plt.xlim([-5,5])
plt.ylabel('Log-likelihood')
plt.xlabel('Distance (mm)')
plt.legend(loc='upper left')
# plt.savefig('experiments/high-noise-single-reflector.png')
plt.show()
