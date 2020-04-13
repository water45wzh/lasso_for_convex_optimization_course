function [x, iter, out] = l1_ALM_dual(x0, A, b, mu, opts)
out.index = "3.g";
out.description = "Augmented Lagrangian method for the dual problem";

%initialization of opts
opts.max_iter_schedule = [50,50,50,50,1e+6];
opts.multi_mu_schedule = [1e+4,1e+3,1e+2,1e+1,1];
opts.err = 1e-10;
opts.step_size = 1;
opts.alpha = 1; % slack for w

[m,n] = size(A);
num_epoch = length(opts.multi_mu_schedule);
x_now = x0;
w = zeros(n,1);
z = zeros(m,1);
err = opts.err;
alpha = opts.alpha;
max_iter_schedule = opts.max_iter_schedule;
t = opts.step_size;
l_now = x0;
iter = 1; % number of iteration

inv_z = inv(eye(m) + t*A*A');
inv_w1 = inv(t^2*A'*inv_z*A - (1+alpha)*t*eye(n));
inv_w2 = t*A'*inv_z;

for epoch = 1:num_epoch
    mu0 = mu*opts.multi_mu_schedule(epoch);
    l_prev = 100*ones(n,1);
    k = 1;
    while norm(l_now-l_prev)/(1+norm(l_now)) >= err & k < max_iter_schedule(epoch)
        l_prev = l_now;
        w = inv_w1*(-l_now + inv_w2*(A*l_now + b));
        w = min(mu0, max(w, -mu0));
        z = inv_z*(t*A*w - A*l_now - b);
        l_now = l_now + t*(A'*z - w);
        k = k + 1;
        iter = iter + 1;
    end
end
l = [];
az = A'*z;
A1 = [];
for i = 1:n
    if abs(abs(az(i))-mu0) < 1e-9
        l = [l i];
        A1 = [A1 A(:,i)];
    end
end
x1 = (A1'*A1)\A1'*(z+b);
x = zeros(n,1);
len_l = length(l);


if len_l ~= 0
    for i = 1:len_l
        x(l(i)) = x1(i);
    end
end
        
out.az = az;
out.res.optval = -0.5*z'*z-b'*z;
fprintf('%s has been finished! iter: %d \n \n', out.description, iter)
end

