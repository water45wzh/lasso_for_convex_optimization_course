function [x,iter,out] = l1_gurobi(x0, A, b, mu, opts)
out.index = "2.b";
out.description = "solve it directly by gurobi";

clear model;
[m,n] = size(A);

%rewrite this problem in order to solve it directly by gurobi
Q = zeros(2*n+m,2*n+m);
Q(2*n+1:2*n+m,2*n+1:2*n+m) = 0.5*eye(m);
model.Q = sparse(Q);
model.obj = [zeros(n,1);mu*ones(n,1);zeros(m,1)];
model.A = sparse([eye(n) -eye(n) zeros(n,m); -eye(n) -eye(n) zeros(n,m); A zeros(m,n) -eye(m)]);
model.rhs = [zeros(2*n,1);b];
model.lb = repmat(-inf,2*n+m,1); 
model.sense = [repmat('<',2*n,1);repmat('=',m,1)];
%solve it
results = gurobi(model);
x = results.x(1:n);
iter = -1;
out.res = results;
end