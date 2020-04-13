function [x,iter,out] = l1_PGD_primal(x0, A, b, mu, opts)
out.index = "3.a";
out.description = "Projection gradient method";

%initialization of opts
opts.multi_mu_schedule = [1e+4,1e+3,1e+2,1e+1,1]; % mu times i in this schedule
opts.max_iter_schedule = [500,500,500,500,1e+6]; % maximal number of iteration per epoch
opts.err = 1e-8;

%rewrite this problem to a QP
num_epoch = length(opts.multi_mu_schedule);
n = size(x0,1);
W = [A, -A];
x_now = x0;
err = opts.err;
max_iter_schedule = opts.max_iter_schedule;

iter = 0;
% iteration start
for epoch = 1:num_epoch
    mu0 = mu*opts.multi_mu_schedule(epoch);
    x_prev = 100*ones(n,1);
    k = 1; % number of iteration in the following sub-iteration
    while norm(x_now-x_prev)/(1+norm(x_now)) >= err & k < max_iter_schedule(epoch)
        x_prev = x_now;
        % projection
        x1 = max(x_now,0);
        x2 = -min(x_now,0);
        y = [x1;x2];
        % compute gradient
        g = W'*(W*y-b) + mu0*ones(2*n,1);
        % compute exact step size in this direction
        t = g'*g/(g'*W'*W*g);
        % update y with projection
        y = max(y-t*g, 0);
        % y to x
        x_now = y(1:n) - y(n+1:2*n);
        k = k + 1;
        iter = iter + 1;
    end
end
x = x_now;
%out.res.optval = 0.5*norm(A*x-b,2)+mu*norm(x,1);
out.res.optval = 0.5*(A*x-b)'*(A*x-b)+mu*norm(x,1);
fprintf('%s has been finished! iter: %d \n \n', out.description, iter)
end