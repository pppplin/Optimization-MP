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
lambda = 1;
c = 1000;
% S = hx.^2 + hy.^2;
% sum_S = sum(S,'all');

% create sqrt(l^2 + h_x^2 + h_y^2)
AREA_S = zeros(M);
for i = 1:M
    for j = 1:M
        if i == M
            if j == M
                AREA_S(i, j) = d_l^2 + (vh(i, j) - vh(i-1, j))^2 + (vh(i, j) - vh(i, j-1))^2;
            else
                AREA_S(i, j) = d_l^2 + (vh(i, j) - vh(i-1, j))^2 + (vh(i, j) - vh(i, j+1))^2;
            end
        end

        if j == M
            if i ~= M
                AREA_S(i, j) = d_l^2 + (vh(i+1, j) - vh(i, j))^2 + (vh(i, j) - vh(i, j-1))^2;
            end
        end

        if i < M & j < M
            AREA_S(i, j) = d_l^2 + (vh(i+1, j) - vh(i, j))^2 + (vh(i, j+1) - vh(i, j))^2;
        end
    end
end
ARE_S = sqrt(AREA_S);
sum_AREA_S = sum(AREA_S, 'all') / d_l;

g = zeros(M);

g(1, 1) = (2 * vh(1, 1) - vh(2, 1) - vh(1, 2)) / AREA_S(1, 1);

% i = 1
i = 1;
for j = 2:(M-1)
    g(i, j) = (2 * vh(i, j) - vh(i + 1, j) - vh(i, j + 1)) / AREA_S(i, j);
    g(i, j) = g(i, j) + (vh(i, j) - vh(i, j - 1)) / AREA_S(i, j - 1);
end

% i = M
i = M;
g(M, 1) = (2 * vh(M, 1) - vh(M-1, 1) - vh(M, 2)) / AREA_S(M, 1);
g(M, 1) = g(M, 1) + (vh(M, 1) - vh(M-1, 1)) / AREA_S(M - 1, j);
for j = 2:(M-1)
    g(i, j) = (2 * vh(i, j) - vh(i-1, j) - vh(i, j+1)) / AREA_S(i,j);
    g(i, j) = g(i, j) + (vh(i, j) - vh(i- 1, j)) / AREA_S(i - 1, j);
    g(i, j) = g(i, j) + (vh(i, j) - vh(i, j-1)) / AREA_S(i, j-1);
end

% j = 1
j = 1;
for i = 2:(M-1)
    g(i, j) = (2 * vh(i, j) - vh(i + 1, j) - vh(i, j + 1)) / AREA_S(i, j);
    g(i, j) = g(i, j) + (vh(i, j) - vh(i- 1, j)) / AREA_S(i - 1, j);
end

% j = M
j = M;
g(1, M) = (2 * vh(1, M) - vh(2, M) - vh(1, M-1)) / AREA_S(1, M);
g(1, M) = g(1, M) + (vh(1, M) - vh(1, M-1)) / AREA_S(1, M-1);
for i = 2:(M-1)
    g(i, j) = (2 * vh(i, j) - vh(i+1, j) - vh(i, j-1)) / AREA_S(i, j);
    g(i, j) = g(i, j) + (vh(i, j) - vh(i-1, j)) / AREA_S(i, j);
    g(i, j) = g(i, j) + (vh(i, j) - vh(i, j - 1)) / AREA_S(i, j - 1);
end

g(M, M) = (2 * vh(M, M) - vh(M-1, M) - vh(M, M-1)) / AREA_S(M, M);
g(M, M) = g(M, M) + (vh(M, M) - vh(M - 1, M)) / AREA_S(M-1, M);
g(M, M) = g(M, M) + (vh(M, M) - vh(M, M - 1)) / AREA_S(M, M-1);

for i = 2:(M-1)
    for j = 2:(M-1)
        g(i, j) = (2 * vh(i, j) - vh(i + 1, j) - vh(i, j + 1)) / AREA_S(i, j);
        g(i, j) = g(i, j) + (vh(i, j) - vh(i- 1, j)) / AREA_S(i - 1, j);
        g(i, j) = g(i, j) + (vh(i, j) - vh(i, j - 1)) / AREA_S(i, j - 1);
    end
end

g(i, j) = g(i, j) * 2 / d_l;

constraint = [vh(1,1)-1,vh(1,half),vh(1,half)-1,vh(half,1),vh(half,half)-1,vh(half,M),vh(M,1)-1,vh(M,half),vh(M,M)-1];

H_c = zeros(M);
% H_c(1,1) = (c - lambda)*(vh(1,1) - 1);
% H_c(1,half) = (c - lambda)*(vh(1,half));
% H_c(1,M) = (c - lambda)*(vh(1,M) - 1);
% H_c(half,1) = (c - lambda)*(vh(half,1));
% H_c(half,half) = (c - lambda)*(vh(half,half) - 1);
% H_c(half,M) = (c - lambda)*(vh(half,M));
% H_c(M,1) = (c - lambda)*(vh(M,1) - 1);
% H_c(M,half) = (c - lambda)*(vh(M,half));
% H_c(M,M) = (c - lambda)*(vh(M,M) - 1);
H_c(1,1) = c *(vh(1,1) - 1)- lambda;
H_c(1,half) = c*(vh(1,half)) - lambda;
H_c(1,M) = c*(vh(1,M) - 1) - lambda;
H_c(half,1) = c *(vh(half,1)) - lambda;
H_c(half,half) = c*(vh(half,half) - 1) - lambda;
H_c(half,M) = c*(vh(half,M)) - lambda;
H_c(M,1) = c*(vh(M,1) - 1) - lambda;
H_c(M,half) = c*(vh(M,half)) - lambda;
H_c(M,M) = c*(vh(M,M) - 1)- lambda;

%{
for i = 2:(half-1)
    % (0, 0, 1) --> (0, 0.5, 0)
    constraint = [constraint, vh(1, i) - 1 * (129 - i) / 128];
    H_c(1, i) = c * (vh(1, i) - 1 * (129 - i) / 128) - lambda;
    % (0, 0.5, 0) --> (0, 1, 1)
    constraint = [constraint, vh(1, 128 + i) - 1 * (i - 1) / 128];
    H_c(1, 128 + i) = c * (vh(1, 128 + i) - 1 * (i - 1) / 128) - lambda;
    % (0.5, 0, 0) --> (0.5, 0.5, 1)
    constraint = [constraint, vh(129, i) - 1 * (i - 1) / 128];
    H_c(129, i) = c * (vh(129, i) - 1 * (i - 1) / 128)-lambda;
    % (0,5, 0.5, 1) --> (0.5, 1, 0)
    constraint = [constraint, vh(129, 128 + i) - 1 * (129 - i) / 128];
    H_c(129, 128 + i) = c * (vh(129, 128 + i) - 1 * (129 - i) / 128) -lambda;
    % (1, 0, 1) --> (1, 0.5, 0)
    constraint = [constraint, vh(257, i) - 1 * (129 - i) / 128];
    H_c(257, i) = c * (vh(257, i) - 1 * (129 - i) / 128) - lambda;
    % (1, 0.5, 0) --> (1, 1, 1)
    constraint = [constraint, vh(257, 128 + i) - 1 * (i - 1) / 128];
    H_c(257, 128 + i) = c * (vh(257, 128 + i) - 1 * (i - 1) / 128) - lambda;
    % (0, 0, 1) --> (0.5, 0, 1)
    constraint = [constraint, vh(i, 1) - 1 * (129 - i) / 128];
    H_c(i, 1) = c * (vh(i, 1) - 1 * (129 - i) / 128) - lambda;
    % (0.5, 0, 1) --> (1, 0, 1)
    constraint = [constraint, vh(128 + i, 1) - 1 * (i - 1) / 128];
    H_c(128 + i, 1) = c * (vh(128 + i, 1) - 1 * (i - 1) / 128) - lambda;
    % (0, 0.5, 1) --> (0.5, 0.5, 1)
    constraint = [constraint, vh(i, 129) - 1 * (i - 1) / 128];
    H_c(i, 129) = c * (vh(i, 129) - 1 * (i - 1) / 128) - lambda;
    % (0.5, 0.5, 1) --> (1, 0.5, 0)
    constraint = [constraint, vh(128 + i, 129) - 1 * (129 - i) / 128];
    H_c(128 + i, 129) = c * (vh(128 + i, 129) - 1 * (129 - i) / 128) - lambda;
    % (0, 1, 1) --> (0.5, 1, 0)
    constraint = [constraint, vh(i, 257) - 1 * (129 - i) / 128];
    H_c(i, 257) = c * (vh(i, 257) - 1 * (129 - i) / 128) - lambda;
    % (0.5, 1, 0) --> (1, 1, 1)
    constraint = [constraint, vh(128 + i, 257) - 1 * (i - 1) / 128];
    H_c(128 + i, 257) = c * (vh(128 + i, 257) - 1 * (i - 1) / 128) - lambda;
end
%}

f = sum_AREA_S - lambda * sum(constraint) + (c/2) * sum(constraint.^2);
g = g + H_c;
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
