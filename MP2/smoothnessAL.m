function [f, g] = smoothnessAL(vh)
N = 256;
M = N + 1;
half = ceil( M / 2 );
A_g = 4*eye(M);
A = -eye(M);
for r = 1:N
    A(r,r+1) = 1;
    A_g(r,r+1) = -2;
    A_g(r+1,r) = -2;
end
A_g(1,1) = 6;
A_g(M, M) = 6;
A = N*A;
hx = vh*A;
hy = A'*vh;
% TODO: finetune itialization
lambda = 1; 
c = 100;
S = hx.^2 + hy.^2;
sum_S = sum(S,'all');

constraint = [abs(vh(1,1)-1),abs(vh(1,half)),abs(vh(1,half)-1),abs(vh(half,1)),abs(vh(half,half)-1),abs(vh(half,M)),abs(vh(M,1)-1),abs(vh(M,half)),abs(vh(M,M)-1)];
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
g = N*N*(A_g*vh + vh*A_g) + H_c;
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
