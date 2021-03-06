function [f, g] = smoothnessAL(vh)
N = 256;
M = N + 1;
half = ceil( M / 2 );
A_g = 36*eye(M);
A = -2*eye(M);
for r = 1:N
    A(r,r+1) = 1;
    A(r+1,r) = 1;
    A_g(r,r+1) = -24;
    A_g(r+1,r) = -24;
end
for r = 1:N-1
    A_g(r,r+2) = 6;
    A_g(r+2,r) = 6;
end

A(1,1) = 1;
A(2,1) = -1;
A(M,M) = 1;
A(M-1,M) = -1;
A_g(1,1) = 12;
% A_g(2,1) = 30;
% A_g(3,1) = 30;
A_g(1,2) = -18;
A_g(2,1) = -18;
A_g(M,M) = 12;
% A_g(2,2) = 10;
% A_g(M-1,M-1) = 10;
A_g(M-1,M) = -18;
A_g(M,M-1) = -18;

A = N*A;
hx = vh*A;
hy = A'*vh;
% TODO: finetune itialization
lambda = 1; 
c = 100000;
S = hx.^2 + hy.^2;
sum_S = sum(S,'all');

constraint = [vh(1,1)-1,vh(1,half),vh(1,half)-1,vh(half,1),vh(half,half)-1,vh(half,M),vh(M,1)-1,vh(M,half),vh(M,M)-1];
H_c = zeros(N+1);
H_c(1,1)= (c - lambda)*(vh(1,1) - 1);
H_c(1,half)= (c - lambda)*(vh(1,half));
H_c(1,M)= (c - lambda)*(vh(1,M) - 1);
H_c(half,1)= (c - lambda)*(vh(half,1));
H_c(half,half)= (c - lambda)*(vh(half,half) - 1);
H_c(half,M)= (c - lambda)*(vh(half,M));
H_c(M,1)= (c - lambda)*(vh(M,1) - 1);
H_c(M,half)= (c - lambda)*(vh(M,half));
H_c(M,M)= (c - lambda)*(vh(M,M) - 1);

f = sum_S - lambda*sum(constraint) + (c/2)*sum(constraint.^2);
g = N*N*(A_g'*vh + vh*A_g) + H_c;
% if nargout > 1 % gradient required
%     % g = 4A'AH; Hessian = 4A'A
%  
%     if nargout > 2 % Hessian required
%         % TODO: approximation should be the following, but exactly how???
%         % https://www.mathworks.com/help/optim/ug/fminunc.html#butpb7p-options
%         % USE HessianMultiplyFcn?? 
%         H = 4*sparse(A')*sparse(A);  
%     end
% end
end
