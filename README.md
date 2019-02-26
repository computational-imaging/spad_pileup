# Data and MATLAB code used in "Sub-picosecond photon-efficient 3D imaging using single-photon sensors"

The methods implemented in this code use photon arrival data obtained by a SPAD sensor to compute accurate 3D geometry and reflectivity of small objects as a proof-of-concept (bounded by a 20cm x 20cm x 20cm volume).

The experimental results from the submitted manuscript and simulations from the supplemental material are reproducible with this implementation. The proposed probabilistic reconstruction method, and various baseline methods, are implemented completely in MATLAB code, without dependency on third-party libraries. While the run time performance is orders of magnitude larger than our corresponding GPU implementation, this code ensures high portability and thereby ease of reproducibility. All demonstrated reconstructions use only three mixture components for the impulse model, which makes the method implementation substantially more accessible, while only minimally affecting result quality. 

The folder 'comparisons' contains code for comparisons of the proposed methods against various baselines discussed in the submitted manuscript and supplemental material. We have included a simulation scene dataset to validate the method in simulation. The entry point script is 'simulation_probabilistic_2D.m'. 

The folder 'experiments' contains code for reproducing the results of the proposed method on the datasets from the submitted manuscript. The entry point launch script is 'reconstruction_probabilistic_2D.m'. 

This code has been tested on MATLAB 2017a under Linux, on a 2.4GHz notebook computer with 16Gb RAM.

# Troubleshooting:

*Code breaks on 'pcshow' in 'reconstruction_probabilistic_2D': Upgrade to MATLAB >= R2017a or set 'display_pointclouds = false';
*Cannot launch parallel pool error: Set 'gpu_implem = false';
*Legacy machine/No GPU error in 'reconstruction_probabilistic_2D': Set 'quick_compute = true'.
*Code runtime very long (no parallel compute) in 'reconstruction_probabilistic_2D': Set 'quick_compute = true'.
