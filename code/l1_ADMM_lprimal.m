function [x,iter,out] = l1_ADMM_lprimal(x0, A, b, mu, opts)
out.index = "3.i";
out.description = "Alternating direction method of multipliers with linearization for the primal problem";

%initialization of opts
opts.multi_mu_schedule = [1e+4,1e+3,1e+2,1e+1,1]; % mu times i in this schedule
opts.max_iter_schedule = [50,50,50,50,1e+6];
opts.err = 1e-8;

num_epoch = length(opts.multi_mu_schedule);
[m,n] = size(A);
x_now = x0;
l = zeros(n,1);
w = zeros(n,1);
err = opts.err;
max_iter_schedule = opts.max_iter_schedule;
t = 2;

L = chol(t*eye(n)+A'*A,'lower');
L = sparse(L);
U = sparse(L');

iter = 0;
% iteration start
for epoch = 1:num_epoch
    mu0 = mu*opts.multi_mu_schedule(epoch);
    st = mu0/t;
    x_prev = 100*ones(n,1);
    k = 1; % number of iteration in the following sub-iteration
    %a = step_size_schedule(epoch);
    while norm(x_now-x_prev)/(1+norm(x_now)) >= err & k < max_iter_schedule(epoch)
        ee = norm(x_now-x_prev)/(1+norm(x_now));
        x_prev = x_now;
        x_now = U\(L\(A'*b + t*w - l));
        tmp = x_now + l/t;
        w = zeros(n,1);
        w(tmp>st) = tmp(tmp>st)-st;
        w(tmp<-st) = tmp(tmp<-st)+st;
        l = l + t*(x_now-w);
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



