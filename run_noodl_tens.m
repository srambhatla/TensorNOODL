% Main TensorNOODL

% factor A - n x m 
% factor B - J x m 
% factor C - K x m 
% a -- alpha
% eta -- step size for A update
% seed -- the random seed (we used 42, 26, and 91)
% out_folder -- the folder where the result of each run will be saved (as a
% .mat file)
function[done] = run_noodl_tens(n, m, J, a, eta, seed, out_folder)
display('Welcome, Lets make some TensorNoodl(s).....')
 
% Uncomment and define if not running on a HPC resource.
% myCluster = parpool('local',4);

% Set Random seed
rng(seed)

verbose = 1;
success = 0;
%% Parameters
clear all 
% Data Generation 

% Uncomment the following for a quick test!
% n = 100;  
% m = 150; 
% J = 100;
% a = 0.05; 
% eta = 20;
% b = a; %(beta and alpha)
% out_folder = './'

b = a; 
K = J;



% IHT parameters
c = 0.2; thr_c = 0.1; % c = step size for the IHT step, thr_c is the threshold

% Set step sizes of other algorithms as the one used by TensorNOODL
eta_arora = eta;
eta_arora_red = eta;

tol_X = 1e-12; % Target tolerance for the recovered X
tol_A = 1e-10; % Target tolerance for the recovered A

%% Generate Data 

display('O. Generate Data .....')

% Dicitionary A_o (_o denotes original analogous to * in the paper)
A_o = randn(n, m); A_o = nrmc(A_o);
display(['Incoherence parameter (mu) = ', num2str(max(max(abs(A_o'*A_o)-eye(m))))])


%% Prep TensorNoodl : Initialize dictionary
display('O. Initialize Dictionary Factor Randomly .....')

noise =  nrmc(randn(size(A_o))); noise = 2*noise*(1/log(m));
A = A_o + noise; A = nrmc(A);

%% Make TensorNoodl: Alternating Minimization to refine the Dictionary: distributed
show = 1;

close all
tic;
try 

[A_our, A_arora, A_arora_red, A_odl, errA, errX, err,  err_arora, errA_arora, errX_arora, err_arora_red, ...
    errA_arora_red, errX_arora_red, errA_odl, errX_odl, err_odl, Y_last, X_last, X_last_o, X_arora_last, X_arora_red_last, X_odl_last, ...
    time_our, time_arora_red, time_arora, time_odl]...
    = tens_compare_NOODL(A, A_o, J, K, a, b, c, thr_c, eta, eta_arora, eta_arora_red, tol_X, tol_A, out_folder, show);
catch ME
  if strcmp(ME.identifier, 'TensorDecomp:DataMatZero')
       warning('Low probabilities alpha and beta resulted in a zero data matrix.')
  else 
     rethrow(ME)
  end
end
learn_time = toc;
display(['TensorNoodl is ready! It took ', num2str(learn_time/60), ' minutes to make TensorNoodls.'])
done = 1

 
%% Descriptions of the outputs
% X_last_o -- the ground truth X matrix at the last iteration
% Y_last -- the corresponding data matrix at the last iteration 

% TensorNOODL
% A_our -- TensorNOODL estimate of factor A
% X_last -- the final estimate of X_last_o
% errA -- trajectory of the error in factor A
% errX -- trajectory of the error in X (sparse matrix)
% err -- trajectory of fit error
% time_our -- time taken by TensorNOODL at each iteration

% Arora (biased)
% A_arora -- Arora(biased) estimate of factor A
% errA_arora -- trajectory of the error in factor A
% errX_arora -- trajectory of the error in X (sparse matrix)
% err_arora -- trajectory of fit error
% time_arora -- time taken at each iteration

% Arora (unbiased)
% A_arora_red -- Arora(unbiased) estimate of factor A
% errA_arora_red -- trajectory of the error in factor A
% errX_arora_red -- trajectory of the error in X (sparse matrix)
% err_arora_red -- trajectory of fit error
% time_arora_red -- time taken at each iteration

% Mairal's Online Dictionary Learning (ODL)
% A_odl -- Mairal's estimate of factor A
% errA_odl -- trajectory of the error in factor A
% errX_odl -- trajectory of the error in X (sparse matrix)
% err_odl -- trajectory of fit error
% time_odl -- time taken at each iteration

