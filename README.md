# TensorNOODL
NOODL for Structured Tensor Factorization

## How to make TensorNOODLs!

The results shown in this work can be reproduced using the attached code. This code was run on a HP Haswell Linux Cluster. Specifically, we used 20 cores and a total of 200 GB RAM; see Appendix E in the supplementary material accompanying the manuscript.

This implementation is in MATLAB, the code will scale according to the number of workers made available. We highly recommend starting with managable sizes of J, K, alpha, beta, and m, so as to not end-up with an unresponsive system. 

Note that in this implementation we set alpha = beta, and J = K, in order to narrow down the scope of experimental study. However, the code can be modified to consider other cases. 

The random seeds used for Monte-Carlo simulations were 42, 26, and 91. 

## Function Dependence

run_noodl_tens.m (the wrapper function)
|-----tens_compare_NOODL.m (compare the results with other techniques)
|      |------best_fista_result.m (recovers the estimates of the sparse matrix X)
|              |------FISTA_with_init.m (implementation of FISTA)
|              |------FISTA_with_init_stochastic.m (stochastic version of ISTA)
|-----softThr.m (the soft thresholding function for FISTA and ISTA)
|-----KRP.m (evaluates the Khatri-Rao Product)
|-----nrmc.m (normalizes the columns of a matrix)               

Note: if only evaluating TensorNOODL's performance, the code corresponding to other techniques can be commented out in the tens_compare_NOODL.m  file. 


## Step-size Recommendations 

Step Size (dictionary Update): We used the following Step sizes for dictionary update (eta in code eta_A in paper), depending upon the rank m.

m = 50 eta = 20 (except for alpha = 0.005 where eta = 5)
m = 150, eta = 40
m = 300, eta = 40
m = 450, eta = 50
m = 600, eta = 50

