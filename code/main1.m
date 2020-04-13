% function Test_l1_regularized_problems

% min 0.5 ||Ax-b||_2^2 + mu*||x||_1

% generate data
clear all
seed = 97006855;
fprintf('rand_seed=%d;\n', seed);
ss = RandStream('mt19937ar', 'Seed', seed);
RandStream.setGlobalStream(ss);

n = 1024;
m = 512;
clear A u b;
A = randn(m,n);
u = sprandn(n,1,0.1);
b = A*u;

mu = 1e-3;

x0 = rand(n,1);

errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));

% Problem 1
% cvx calling mosek
opts1 = []; %modify options
tic; 
[x1, iter1, out1] = l1_cvx_mosek(x0, A, b, mu, opts1);
t1 = toc;
sp1 = get_sparsity(x1);

% cvx calling gurobi
opts2 = []; %modify options
tic; 
[x2, iter2, out2] = l1_cvx_gurobi(x0, A, b, mu, opts2);
t2 = toc;
sp2 = get_sparsity(x2);

% Problem 2
% call mosek directly
opts3 = []; %modify options
tic; 
[x3, iter3, out3] = l1_mosek(x0, A, b, mu, opts3);
t3 = toc;
sp3 = get_sparsity(x3);

% call gurobi directly
opts4 = []; %modify options
tic; 
[x4, iter4, out4] = l1_gurobi(x0, A, b, mu, opts4);
t4 = toc;
sp4 = get_sparsity(x4);

% Problem 3
% 3.a PGD
opts5 = []; %modify options
tic; 
[x5, iter5, out5] = l1_PGD_primal(x0, A, b, mu, opts5);
t5 = toc;
sp5 = get_sparsity(x5);

% 3.b SGD
opts6 = []; %modify options
tic; 
[x6, iter6, out6] = l1_SGD_primal(x0, A, b, mu, opts6);
t6 = toc;
sp6 = get_sparsity(x6);

% print comparison results with cvx-call-mosek
fprintf('cvx-call-gurobi: cpu: %5.2f, iter: %d, err-to-cvx-mosek: %3.2e\n', t2, iter2, errfun(x1, x2));
fprintf('     call-mosek: cpu: %5.2f, iter: %d, err-to-cvx-mosek: %3.2e\n', t3, iter3, errfun(x1, x3));
fprintf('    call-gurobi: cpu: %5.2f, iter: %d, err-to-cvx-mosek: %3.2e\n', t4, iter4, errfun(x1, x4));
fprintf('            PGD: cpu: %5.2f, iter: %d, err-to-cvx-mosek: %3.2e\n', t5, iter5, errfun(x1, x5));
fprintf('            SGD: cpu: %5.2f, iter: %d, err-to-cvx-mosek: %3.2e\n', t6, iter6, errfun(x1, x6));

function s = get_sparsity(x)
%evaluate sparsity of a vector with threshold 1e-9
n = length(x);
th = 1e-9;
count = 0;
for i = 1:n
    if abs(x(i)) < th
        count = count + 1;
    end
end
s = count/n;
end