function [A_our, A_arora, A_arora_red, A_odl, errA, errX, err,  err_arora, errA_arora, errX_arora, ...
    err_arora_red, errA_arora_red, errX_arora_red, errA_odl, errX_odl, err_odl,Y_last, X_last, X_last_o, X_arora_last, X_arora_red_last, X_odl_last, ...
    time_our, time_arora_red, time_arora, time_odl]...
    = tens_compare_NOODL(A, A_o, J1, K1, a, b, c, thr_c, eta, eta_arora, eta_arora_red, tol_X, tol_A, out_folder, show)

display('II. Making TensorNoodl: Run the TensorNoodl Algorithm .....')

rand(2,2) % checking seeds

n = size(A,1);
m = size(A,2);
k_x = ceil(m*a*b); k = k_x; % number of nonzeros, used to determine is there will be enough samples for learning

% Parameters for generating coefficient matrices 
C = 1; % Non-zero sparse elements generated from Rademacher distribution

% Initialize all factor As
A_our = A;
A_arora = A;
A_arora_red = A;
A_odl = A;

errA = []; errA_arora = []; errA_arora_red = []; errA_odl = [];
errX = []; errX_arora = []; errX_arora_red = []; errX_odl = [];
err = []; err_arora = []; err_arora_red = []; err_odl = [];

time_our = [];
time_arora_red = [];
time_arora = [];
time_odl = [];
time_odl_coeff = [];

% ODL Specific
E = zeros(m, m);
F = zeros(n, m);

i = 1;
change_A = norm(A - A_o, 'fro')/norm(A_o,'fro');

if show
figure('pos',[1000 1500 1200 600])
end
last = 0;
display('Begin comparing...')
while((change_A>tol_A))   
    
  %% Data Generation for all algorithms
  
  % Generate new sample
  B_new = sparse(sign(randn(J1,m).*binornd(1,a,J1,m)));
  C_new = sparse(sign(randn(K1,m).*binornd(1,b,K1,m))); 

  % Khatri Rao Product 
  X_new_whole = KRP(C_new,B_new)'; 
  X1_new = A_o*X_new_whole; 
  non_zeros = nnz(X1_new);  


  % Check if the data is all zero!
  if ~nnz(X_new_whole)
     tr = 1;
      while((~nnz(X_new_whole)) && (tr<10))
           % Generate new sample
            B_new = sparse(sign(randn(J1,m).*binornd(1,a,J1,m)));
            C_new = sparse(sign(randn(K1,m).*binornd(1,b,K1,m))); 

            % Khatri Rao Product 
            X_new_whole = KRP(C_new,B_new)'; 
            X1_new = A_o*X_new_whole; 
            non_zeros = nnz(X1_new); 
            tr = tr + 1;
      end
      if ~nnz(X_new_whole)
            name = strcat(out_folder,'res_tens_n_',num2str(n),'_m_',num2str(m),'_J_',... 
            strrep(num2str(J1),'.','_'),'_etaA_',strrep(num2str(eta),'.','_'),'_alpha_',strrep(num2str(a), '.','_'),'.mat')
            save(name,'non_zeros')

            ME = MException('TensorDecomp:DataMatZero', 'Non-zeros in B and C are too low, resulted in a zero matrix!');
            throw(ME)  
      end
    
  end

  fi = X1_new; % the fibers 
  idx = find(sum(abs(fi),1));  
  Y_m = fi(:, idx);     
  X_new = X_new_whole(:,idx); 
  pf = size(Y_m,2) ; % the number of data samples
  
  %% IHT stage (TensorNOODL) 
  % The code decides if it needs to use MATLAB's spmd (distributed computing) or 
  % if the data is managable for non-distributed settings.
   
   if (pf > 1000)
      tic
      % Hard Thresholding
      AtY = A_our'*Y_m;
      XS = AtY.*(abs(AtY)>=C/2);
  
      % Error after HT
      AtA_our = A_our'*A_our;
      
      % Prep for Matlab's spmd
      clear change_X change_X_g
      change_X_g= Composite();
      err_int_X_g = Composite();
      for lab = 1:length(change_X_g)
         change_X_g{lab} = 1;
         err_int_X_g{lab} = [];
      end
      ii = 0;   
      
      spmd
   
      XS_dist = codistributed(XS);
      X_new_dist = codistributed(X_new, getCodistributor(XS_dist));
      Y_m_dist = codistributed(Y_m, getCodistributor(XS_dist));
       
      change_X = codistributed(change_X_g);
      err_int_X = codistributed(err_int_X_g);
      
      local_XS = getLocalPart(XS_dist);
      dist = getCodistributor(XS_dist);
      local_X_new = getLocalPart(X_new_dist);
      local_Y_m = getLocalPart(Y_m_dist);
    
      err_int_X =  norm(local_XS(:) - local_X_new(:))/norm(local_X_new(:));
   
       % IHT 
         while(change_X > tol_X)
            eta_a = c;
            thr = thr_c;

            local_XS = local_XS - eta_a*((AtA_our*local_XS) - A_our'*local_Y_m );
            local_XS(abs(local_XS) <= thr) = 0;
 
            err_int_X = [err_int_X norm(local_XS(:) - local_X_new(:))/norm(local_X_new(:))];
            if (ii>2)
            change_X = abs(err_int_X(end) - err_int_X(end-2));
            end
        
           
            if(show)
                subplot(121)
                semilogy(err_int_X)
                title({'Convergence of Current Coefficient Learning Step', ['n = ', num2str(n), ' m = ', num2str(m), ' k = ', num2str(k_x)]})
                ylabel('Relative error in current coefficient estimate')
                xlabel('Iterations')
                set(gca, 'FontSize',16)
                drawnow
            end
   
            ii = ii + 1;
          
         end
         
      XS_dist = codistributed.build(local_XS, dist, 'noCommunication');
      X_new_dist = codistributed.build(local_X_new, dist, 'noCommunication');
      Y_m_dist = codistributed.build(local_Y_m, getCodistributor(Y_m_dist), 'noCommunication');
      labBarrier;
      end
      
      % Gather results of spmd (result of IHT)
      XS = gather(XS_dist);
      X_new = gather(X_new_dist);
      Y_m = gather(Y_m_dist);
   
   else % Run IHT in non-distributed fashion
       
     tic
     % Hard Thresholding
     AtY = A_our'*Y_m;
     XS = AtY.*(abs(AtY)>=C/2);
     
     AtA_our = A_our'*A_our;
     change_X = 1;
     err_int_X =  norm(XS(:) - X_new(:))/norm(X_new(:));
     ii = 0; 
     
     % IHT 
     while((change_X > tol_X))
         eta_a = c;
         thr = thr_c;
         XS = XS - eta_a*((AtA_our*XS) - A_our'*Y_m );
          
         XS(abs(XS) <= thr) = 0;
         err_int_X = [err_int_X norm(XS(:) - X_new(:))/norm(X_new(:))];
         %plot(err_int_X)
         %drawnow
         if (ii>2)
         change_X = abs(err_int_X(end) - err_int_X(end-2));
         end
         ii = ii + 1;
     end
  end
  
  %% Dictionary Update (TensorNOODL)
  
   % Gradient 
   gr = double(1/pf)*(Y_m - (A_our*XS))*sign(XS)';
   
   % Descend
   A_our = A_our + (eta)*gr;
   A_our = nrmc(A_our);
   t_o = toc;
   
   % Set errors
   err = [err norm((Y_m - A_our*XS),'fro')/norm(Y_m,'fro')]; 
   errA = [errA norm(A_our - A_o,'fro')/norm(A_o,'fro')];
   errX = [errX  norm(XS - X_new,'fro')/norm(X_new,'fro')];
   

   if(isnan(errX(end))||isnan(errA(end)))
       break;
   end
   
   change_A = errA(end);
   
  %% Arora "Unbiased" 
  
   tic
   B = zeros(n,m,m);
   for iii = 1:m
       dict_ele = setdiff([1:m],[iii]);
       B_temp = A_arora_red(:,dict_ele) - (kron((A_arora_red(:,iii)'*A_arora_red(:,dict_ele)),A_arora_red(:,iii)));
       B(:,dict_ele,iii) = B_temp;
       B(:,iii, iii) = A_arora_red(:,iii);
   end
   
   % Hard Thresholding
   AtY = A_arora_red'*Y_m;
   XS_arora_red = AtY.*(abs(AtY)>=C/2);
   gr_arora_red = zeros(size(A));
   for iii = 1:m
       temp_x_i = squeeze(B(:,:,iii))'*Y_m;
       temp_x_i = temp_x_i.*(abs(temp_x_i)>=C/2);
       gr_arora_red(:,iii) =  mean((Y_m - squeeze(B(:,:,iii))*temp_x_i).*kron(sign(XS_arora_red(iii,:)), ones(n,1)), 2);
   end
   
   % Update Dictionary (Descend)
   A_arora_red = A_arora_red + (eta_arora_red)*gr_arora_red;
   A_arora_red = nrmc(A_arora_red);
  
   t_ar1 = toc;
   % Set errors
   err_arora_red = [err_arora_red norm((Y_m - A_arora_red*XS_arora_red),'fro')/norm(Y_m,'fro')];  
   errA_arora_red = [errA_arora_red norm(A_arora_red - A_o,'fro')/norm(A_o,'fro')];
  
   
   %%  Arora "Biased" 

   tic
   % Hard Thresholding
   AtY = A_arora'*Y_m;
   XS_arora = AtY.*(abs(AtY)>=C/2);
   
  
   % Gradient
   gr_arora = (1/pf)*(Y_m - (A_arora*XS_arora))*sign(XS_arora)';

   % Update Dictionary
   A_arora = A_arora + (eta_arora)*gr_arora;
   A_arora = nrmc(A_arora);
   
   t_a1 = toc;

   % Set errors
   err_arora = [err_arora norm((Y_m - A_arora*XS_arora),'fro')/norm(Y_m,'fro')];  
   errA_arora = [errA_arora norm(A_arora - A_o,'fro')/norm(A_o,'fro')];
%% Mairal (ODL)

   % Estimate Coefficients (ISTA or FISTA depending upon the size of the
   % data)
   [XS_odl i_odl t_odl2] = best_fista_result(A_odl, Y_m, A_odl'*Y_m, X_new, tol_X, 20);

   errX_arora_red = [errX_arora_red  norm(XS_arora_red - X_new,'fro')/norm(X_new,'fro')];
   errX_arora = [errX_arora  norm(XS_arora - X_new,'fro')/norm(X_new,'fro')];
   errX_odl = [errX_odl norm(XS_odl - X_new,'fro')/norm(X_new,'fro')];
  
   % Dictionary update 
   tic
   if i < pf
       theta = i*pf;
   else
       theta = pf^2 + i - pf; 
   end
   
   beta = (theta + 1 - pf)/(theta + 1);
    
   E = beta*E + XS_odl*XS_odl';
   F = beta*F + Y_m*XS_odl';
   
   id_mod = setdiff(1:m,find(diag(E)<1e-1)); % ignore diag(E)==0
   
   U = repmat(1./(diag(E(id_mod,id_mod))'),n,1).*(F(:,id_mod) - A_odl(:,id_mod)*E(id_mod,id_mod)) + A_odl(:,id_mod);
   A_odl(:,id_mod) = nrmc(U);
   
   % Set Errors
   errA_odl = [errA_odl norm(A_odl - A_o,'fro')/norm(A_o,'fro')];
   err_odl = [err_odl norm((Y_m - A_odl*XS_odl),'fro')/norm(Y_m,'fro')];   
   t_odl1 = toc;
 
   % Calculate Time 
   t_odl = t_odl1 + t_odl2/i_odl;
   t_ar = t_ar1;
   t_a = t_a1;
   
%% Set time for all algorithms
   time_our = [time_our t_o];
   time_arora_red = [time_arora_red t_ar];
   time_arora = [time_arora t_a];
   time_odl = [time_odl t_odl];
   time_odl_coeff = [time_odl_coeff t_odl2/i_odl];   
   
   
%% Plot
   if(show)

       subplot(122)
       semilogy(errA,  'LineWidth',2)
       hold all
       semilogy(errX,  'LineWidth',2)
       semilogy(err, 'LineWidth',2)
       legend('Error in Dictionary', 'Error in Current Coefficients','Fit Error')
       set(gca, 'FontSize',16)
       grid on
       title({'Convergence of Online Dictionary Learning Algorithm', ['n = ', num2str(n), ' m = ', num2str(m), ' k = ', num2str(k)]})
       ylabel('Cost')
       xlabel('Dictionary Iterations (per fresh sample batch)')
       hold off

       drawnow

   end
  
   display(['TensorNoodl: iter = ', num2str(i), '   ,errA = ', num2str(errA(end)), '   ,errX = ', num2str(errX(end)), ', time = ', num2str(time_our(end))])
   display(['Arora "Unbiased": iter = ', num2str(i), '   ,errA = ', num2str(errA_arora_red(end)), '   ,errX = ', num2str(errX_arora_red(end)),', time = ', num2str(time_arora_red(end))])
   display(['Arora "Biased": iter = ', num2str(i), '   ,errA = ', num2str(errA_arora(end)), '   ,errX = ', num2str(errX_arora(end)), ', time = ', num2str(time_arora(end))])
 display(['ODL Sapiro: iter = ', num2str(i), '   ,errA = ', num2str(errA_odl(end)), '   ,errX = ', num2str(errX_odl(end)), ', time = ', num2str(time_odl(end)), ', time coeff = ', num2str(time_odl_coeff(end))])
   i = i + 1;
  
end

%% Final Coefficient Estimation for Arora(b) and Arora(u)
  spmd
    if(labindex ==1)
         [XS_arora_red_ol, i_ar, t_ar_red_coeff] = best_fista_result(A_arora_red, Y_m, XS_arora_red, X_new, tol_X, 30);
    elseif(labindex ==2)
         [XS_arora_ol, i_a, t_ar_coeff] = best_fista_result(A_arora, Y_m, XS_arora, X_new, tol_X, 30);
    end
  end
  XS_arora_red_ol = XS_arora_red_ol{1};
  XS_arora_ol = XS_arora_ol{2};

  t_ar2 = t_ar_red_coeff{1}/i_ar{1};
  t_a2= t_ar_coeff{2}/i_a{2};


display(['Err Coeff "unbiased" after lasso =',num2str(norm(XS_arora_red_ol - X_new,'fro')/norm(X_new,'fro')),', time coeff', num2str(t_ar_red_coeff{1}),'iter', num2str(i_ar{1})])
display(['Err Coeff "biased" after lasso =',num2str(norm(XS_arora_ol - X_new,'fro')/norm(X_new,'fro')),', time coeff', num2str(t_ar_coeff{2}),'iter coeff', num2str(i_a{2})])

% Set things
 Y_last = Y_m;
 X_last = XS;
 X_last_o = X_new;

 X_arora_last = XS_arora_ol;
 X_arora_red_last = XS_arora_red_ol;
 X_odl_last = XS_odl;
 idx_last = idx;
 
% Save things
name = strcat(out_folder,'res_tens_n_',num2str(n),'_m_',num2str(m),'_J_',...
strrep(num2str(J1),'.','_'),'_etaA_',strrep(num2str(eta),'.','_'),'_alpha_',strrep(num2str(a), '.','_'),'.mat')
save(name, 'A_our', 'A_arora', 'A_arora_red','A_odl', 'errA', 'errX', 'err', 'err_arora', 'errA_arora', 'errX_arora', ...
    'err_arora_red', 'errA_arora_red', 'errX_arora_red', 'errA_odl', 'errX_odl', 'err_odl','Y_last', 'X_last', 'X_last_o', 'X_arora_last','X_arora_red_last','X_odl_last', 'A', 'A_o', 'k_x', 'pf', 'c', 'thr_c', 'eta', 'eta_arora', 'eta_arora_red', ...
    'tol_X', 'tol_A', 'time_our', 'time_arora_red', 'time_arora','time_odl_coeff','time_odl','t_ar2','t_a2', 'idx_last', 'non_zeros')

