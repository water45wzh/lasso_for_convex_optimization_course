function [x,iter,out] = l1_FProxGD_primal(x0, A, b, mu, opts)
out.index = "3.f";
out.description = "Fast proximal gradient method for the primal problem";

%initialization of opts
opts.multi_mu_schedule = [1e+4,1e+3,1e+2,1e+1,1]; % mu times i in this schedule
opts.max_iter_schedule = [5e+2,5e+2,5e+2,5e+2,1e+6];
opts.err = 1e-10;
opts.theta = @(k) 2/(k+1);


num_epoch = length(opts.multi_mu_schedule);
n = size(x0,1);
x_now = x0;
v_now = x0;
err = opts.err;
max_iter_schedule = opts.max_iter_schedule;
t = 1/norm(A,2)^2; % set the fixed step-size
prox = @(x,mu) sign(x) .* max(abs(x)-t*mu,0);

theta = opts.theta;
iter = 0;
% iteration start
for epoch = 1:num_epoch
    mu0 = mu*opts.multi_mu_schedule(epoch);
    x_prev = 100*ones(n,1);
    v_prev = 100*ones(n,1);
    k = 1; % number of iteration in the following sub-iteration
    while norm(x_now-x_prev)/(1+norm(x_now)) >= err & k < max_iter_schedule(epoch)
        ee = norm(x_now-x_prev)/(1+norm(x_now));
        x_prev = x_now;
        v_prev = v_now;
        th = theta(k);
        y = (1-th)*x_prev + th*v_prev;
        % compute gradient
        g = A'*(A*y-b);
        % compute exact step size in this direction
        % update x_now
        x_now = prox(y-t*g,mu0);
        v_now = x_prev + 1/th*(x_now-x_prev);
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

