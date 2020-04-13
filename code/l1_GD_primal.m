function [x,iter,out] = l1_GD_primal(x0, A, b, mu, opts)
out.index = "3.c";
out.description = "Gradient method for the smoothed primal problem";

%initialization of opts
opts.multi_mu_schedule = [1e+4,1e+3,1e+2,1e+1,1]; % mu times i in this schedule
opts.max_iter_schedule = [1e+3,1e+3,1e+3,1e+3,1e+6];
opts.err = 1e-10;
%opts.step_size_schedule = [3*1e-3,3*1e-3,3*1e-3,3*1e-3,3*1e-3];
opts.step_size_schedule = 3e-3;
opts.smoothed = 1e-3; % alpha = smoothed*mu

num_epoch = length(opts.multi_mu_schedule);
n = size(x0,1);
x_now = x0;
err = opts.err;
max_iter_schedule = opts.max_iter_schedule;
step_size_schedule = opts.step_size_schedule;

s_grad = @(x,alpha) sign(x).*min(abs(x)/alpha, 1);

iter = 0;
% iteration start
for epoch = 1:num_epoch
    mu0 = mu*opts.multi_mu_schedule(epoch);
    alpha = mu0*opts.smoothed;
    x_prev = 100*ones(n,1);
    k = 1; % number of iteration in the following sub-iteration
    %a = step_size_schedule(epoch);
    a = step_size_schedule;
    while norm(x_now-x_prev)/(1+norm(x_now)) >= err & k < max_iter_schedule(epoch)
        ee = norm(x_now-x_prev)/(1+norm(x_now));
        x_prev = x_now;
        % compute gradient
        g = A'*(A*x_now-b) + mu0*s_grad(x_now,alpha);
        % compute exact step size in this direction
        t = a/sqrt(k);
        % update x_now
        x_now = x_now-t*g;
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

