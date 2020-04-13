function [x,iter,out] = l1_mosek(x0, A, b, mu, opts)
out.index = "2.a";
out.description = "solve it directly by mosek";

[m,n]=size(A);
%rewrite this problem in order to solve it directly by mosek
Q=zeros(2*n+m,2*n+m);
Q(2*n+1:2*n+m,2*n+1:2*n+m)=eye(m);
Q=sparse(Q);
c=[zeros(n,1);mu*ones(n,1);zeros(m,1)];
L=[eye(n) -eye(n) zeros(n,m); -eye(n) -eye(n) zeros(n,m); A zeros(m,n) -eye(m)];
L=sparse(L);
blc=[-inf(2*n,1);b];
buc=[zeros(2*n,1);b];
%solve it
res=mskqpopt(Q,c,L,blc,buc,[],[]);
%get solution solved by iterior method
x=res.sol.itr.xx(1:n);
iter = -1;
out.res=res.sol.itr;
end