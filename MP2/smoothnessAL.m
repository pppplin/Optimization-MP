function [f,g] = smoothnessAL(H)
N = 256;
A = -eye(N+1);
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

% providing gradient not working
if nargout > 1 % gradient required
    % g = 4A'AH; Hessian = 4A'A
    g = sum(4*sparse(A'*A));  
    if nargout > 2 % Hessian required
        H = [1200*x(1)^2-400*x(2)+2, -400*x(1);
            -400*x(1), 200];  
    end
end
end
