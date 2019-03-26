function [f,g] = smoothnessAL(H)
N = 256;
A = -eye(N+1, N+1);
for r = 1:N
    A(r,r+1) = 1;
end
A = (1/N)*A;
hx = A*H;
hy = H*A';
lambda = 1;
c = 1;
M = hx'*hx + hy'*hy;
h = abs(H(1,1)-1)+abs(H(1,129))+abs(H(1,257)-1)+abs(H(129,1))+abs(H(129,129)-1)+abs(H(129,257))+abs(H(257,1)-1)+abs(H(257,129))+abs(H(257,257)-1);
f = sum(M(:)) - lambda*h + (c/2)*(h*h);

if nargout > 1 % gradient required
    g = sum(4*sparse(A'*A));  
end
end
