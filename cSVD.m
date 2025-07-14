 function [U, S, V] = cSVD(X, k)
% The method in [Benjamin Erichson et al., 2017] paper using Gaussian matrix
s = 7;% S=5
kn = k+s;% Step-01
[~,n] = size(X);
F = randn(n, kn);% Step-02 random test matrix- Parallelism
Y = X*F;% Step-03 sketch input matrix- Parallelism
B = Y'*Y;% Step-04 form smaller lxl matrix- Parallelism
B = 0.5*(B+B');% Step-05 ensure symmetry- Parallelism
[V, D] = eig(B);% Step-06 truncate eigen decomposition
d = diag(D);
e = sqrt(abs(d));% Step-07 rescale eigenvalues
S = spdiags(e, 0, kn, kn);   
U = S\(Y*V)';% Step-08 approximate right singular values- Parallelism
Y = U';    
T = X'*Y;% Step-09 approximate unscaled left singular values- Parallelism
[V, S, U] = svd(T, 'econ');% Step-10 update left singular vectors and values
U = Y*U(:, 1:k);% Step-11 update right singular vectors- Parallelism
S = S(1:k, 1:k);
V = V(:, 1:k);
end