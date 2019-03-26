function [f, g, H] = smoothnessAL(vh)
N = 256;
A = -eye(N+1);
for r = 1:N
    A(r,r+1) = 1;
end
A = N*A;
hx = A*vh;
hy = vh*A';
% initialization??
lambda = 1; 
c = 100;
M = hx'*hx + hy'*hy;
constraint = [abs(vh(1,1)-1),abs(vh(1,129)),abs(vh(1,257)-1),abs(vh(129,1)),abs(vh(129,129)-1),abs(vh(129,257)),abs(vh(257,1)-1),abs(vh(257,129)),abs(vh(257,257)-1)]
f = sum(M(:)) - lambda*sum(constraint) + (c/2)*sum(constraint.^2);

% providing gradient not working
if nargout > 1 % gradient required
    % g = 4A'AH; Hessian = 4A'A
    g = sum(4*sparse(A')*sparse(A)*vh);  
    if nargout > 2 % Hessian required
        H = sum(4*sparse(A')*sparse(A));  
    end
end
end
