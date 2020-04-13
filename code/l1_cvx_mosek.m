function [x, iter, out] = l1_cvx_mosek(x0, A, b, mu, opts)
out.index = "1.a";
out.description = "solve it using cvx by calling mosek";

n=size(A,2);
cvx_solver mosek
cvx_begin quiet
    variable x(n)
    %minimize(0.5*(A*x-b)'*(A*x-b)+mu*sum(abs(x)))
    minimize 0.5*norm(A*x-b,2)+mu*norm(x,1)
cvx_end
iter = -1;
out.res = cvx_optval;
end