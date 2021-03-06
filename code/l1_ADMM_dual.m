function [x,iter,out] = l1_ADMM_dual(x0, A, b, mu, opts)
out.index = "3.h";
out.description = "Alternating direction method of multipliers for the dual problem";

%initialization of opts
opts.multi_mu_schedule = [1e+4,1e+3,1e+2,1e+1,1]; % mu times i in this schedule
opts.max_iter_schedule = [50,50,50,50,1e+6];
opts.err = 1e-8;

num_epoch = length(opts.multi_mu_schedule);
[m,n] = size(A);
x_now = x0;
z = zeros(m,1);
w = zeros(n,1);
err = opts.err;
max_iter_schedule = opts.max_iter_schedule;
t = 3;

L = chol(speye(m)+t*A*A','lower');
L = sparse(L);
U = sparse(L');

iter = 0;
% iteration start
for epoch = 1:num_epoch
    mu0 = mu*opts.multi_mu_schedule(epoch);
    x_prev = 100*ones(n,1);
    k = 1; % number of iteration in the following sub-iteration
    %a = step_size_schedule(epoch);
    while norm(x_now-x_prev)/(1+norm(x_now)) >= err & k < max_iter_schedule(epoch)
        ee = norm(x_now-x_prev)/(1+norm(x_now));
        x_prev = x_now;
        w = A'*z-x_now/t;
        w(w>mu) = mu;
        w(w<-mu) = -mu;
        z = U\(L\(A*x_now-b+t*A*w));
        x_now = x_now + t*(w-A'*z);
        k = k + 1;
        iter = iter + 1;
    end
    fprintf('epoch: %d, sub-iter: %d ,err: %3.2e \n', epoch, k, ee)
end
x = x_now;
%out.res.optval = 0.5*norm(A*x-b,2)+mu*norm(x,1);
out.res.optval = 0.5*(A*x-b)'*(A*x-b)+mu*norm(x,1);
fprintf('%s has been finished! iter: %d \n \n', out.description, iter)
end

