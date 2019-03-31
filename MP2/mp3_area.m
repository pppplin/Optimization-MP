function [f, g] = mp3_area(vh)
N = 256;
M = N + 1;
d_l = 1 / 256;
half = ceil( M / 2 );
% A_g = 4*eye(M);
% A = -eye(M);
% for r = 1:N
%     A(r,r+1) = 1;
%     A_g(r,r+1) = -2;
%     A_g(r+1,r) = -2;
% end
% A_g(1,1) = 2;
% A_g(M,M) = 2;
% A = N*A;
% hx = vh*A;
% hy = A'*vh;
% % TODO: finetune itialization
% lambda = 1;
% c = 100000;
% S = hx.^2 + hy.^2;
% sum_S = sum(S,'all');

% create sqrt(l^2 + h_x^2 + h_y^2)
AREA_S = zeros(M);
for i = 1:M
    for j = 1:M
        AREA_S[i, j] = d_l^2 + (vh[i, j] - vh[i-1, j])^2 + (vh[i, j] - vh[i, j-1])^2
    end
end
ARE_S = sqrt(AREA_S);
sum_AREA_S = sum(AREA_S, 'all') / d_l;

g = zeros(M);
for i = 1:M
    for j = 1:M
        g[i, j] = (2 * vh[i, j] - vh[i + 1, j] - vh[i, j + 1]) / AREA_S[i, j];
        g[i, j] += (vh[i, j] - vh[i- 1, j]) / AREA_S[i - 1, j];
        g[i, j] += (vh[i, j] - vh[i, j - 1]) / AREA_S[i, j - 1];
        g[i, j] = g[i, j] * 2 / d_l;
    end
end

constraint = [vh(1,1)-1,vh(1,half),vh(1,half)-1,vh(half,1),vh(half,half)-1,vh(half,M),vh(M,1)-1,vh(M,half),vh(M,M)-1];

H_c = zeros(M);
H_c(1,1) = (c - lambda)*(vh(1,1) - 1);
H_c(1,half) = (c - lambda)*(vh(1,half));
H_c(1,M) = (c - lambda)*(vh(1,M) - 1);
H_c(half,1) = (c - lambda)*(vh(half,1));
H_c(half,half) = (c - lambda)*(vh(half,half) - 1);
H_c(half,M) = (c - lambda)*(vh(half,M));
H_c(M,1) = (c - lambda)*(vh(M,1) - 1);
H_c(M,half) = (c - lambda)*(vh(M,half));
H_c(M,M) = (c - lambda)*(vh(M,M) - 1);

for i = 1:(half - 1)
    % (0, 0, 1) --> (0, 0.5, 0)
    constraint = [constraint, vh[1, i] - 1 * (129 - i) / 128];
    H_c[1, i] = (c - lambda) * vh[1, i];
    % (0, 0.5, 0) --> (0, 1, 1)
    constraint = [constraint, vh[1, 129 + i] - 1 * (i - 1) / 128];
    H_c[1, 129 + i] = (c - lambda) * vh[1, 129 + i];
    % (0.5, 0, 0) --> (0.5, 0.5, 1)
    constraint = [constraint, vh[129, i] - 1 * (i - 1) / 128];
    H_c[129, i] = (c - lambda) * vh[129, i];
    % (0,5, 0.5, 1) --> (0.5, 1, 0)
    constraint = [constraint, vh[129, 129 + i] - 1 * (129 - i) / 128];
    H_c[129, 129 + i] = (c - lambda) * vh[129, 129 + i];
    % (1, 0, 1) --> (1, 0.5, 0)
    constraint = [constraint, vh[257, i] - 1 * (129 - i) / 128];
    H_c[257, i] = (c - lambda) * vh[257, i];
    % (1, 0.5, 0) --> (1, 1, 1)
    constraint = [constraint, vh[257, 129 + i] - 1 * (i - 1) / 128];
    H_c[257, 129 + i] = (c - lambda) * vh[257, 129 + i];
    % (0, 0, 1) --> (0.5, 0, 1)
    constraint = [constraint, vh[i, 1] - 1 * (129 - i) / 128];
    H_c[i, 1] = (c - lambda) * vh[i, 1];
    % (0.5, 0, 1) --> (1, 0, 1)
    constraint = [constraint, vh[129 + i, 1] - 1 * (i - 1) / 128];
    H_c[129 + i, 1] = (c -lambda) vh[129 + i, 1];
    % (0, 0.5, 1) --> (0.5, 0.5, 1)
    constraint = [constraint, vh[i, 129] - 1 * (i - 1) / 128];
    H_c[i, 129] = (c -lambda) vh[i, 129];
    % (0.5, 0.5, 1) --> (1, 0.5, 0)
    constraint = [constraint, vh[129 + i, 129] - 1 * (129 - i) / 128];
    H_c[129 + i, 129] = (c -lambda) vh[129 + i, 129];
    % (0, 1, 1) --> (0.5, 1, 0)
    constraint = [constraint, vh[i, 257] - 1 * (129 - i) / 128];
    H_c[i, 257] = (c -lambda) vh[i, 257];
    % (0.5, 1, 0) --> (1, 1, 1)
    constraint = [constraint, vh[129 + i, 257] - 1 * (i - 1) / 128];
    H_c[129 + i, 257] = (c -lambda) vh[129 + i, 257];
end

f = sum_S - lambda * sum(constraint) + (c/2) * sum(constraint.^2);
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
