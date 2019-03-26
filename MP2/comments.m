
% fminunc for matrix: https://www.mathworks.com/help/optim/ug/matrix-arguments.html

% interpolation? https://edoras.sdsu.edu/doc/matlab/techdoc/ref/interp2.html
% fminunc must be scaler, change g to scalar. can't use interpolation directly
% x = 0:0.5:1;
% y = 0:0.5:1;
% t = [1,0,1;0,1,0;1,0,1];
% mesh(x,y,t);
% x1=0:(1/N):1;
% y1=0:(1/N):1;
% [x2,y2]=meshgrid(x1,y1);
% t1=interp2(x,y,t,x2,y2,'cubic');

% interpolation
% Hp(1,1) = 1;
% Hp(1,129) = 0;
% Hp(1,257) = 1;
% Hp(129,1) = 0;
% Hp(129,129) = 1;
% Hp(129,257) = 0;
% Hp(257,1) = 1;
% Hp(257,129) = 0;
% Hp(257,257) = 1;

%include hessian: https://www.mathworks.com/help/optim/ug/writing-scalar-objective-functions.html#bu2w6a9-1
if nargout > 1 % gradient required
    g = sum(2*sparse(sparse(A')*sparse(A)));  
    
    if nargout > 2 % Hessian required
        H = sum(2*sparse(sparse(A')*sparse(A)));  
    end

end


N = 256;
H = zeros(N+1, N+1);
A = zeros(N+1, N+1);
Hp = H;

H(1,1) = 1;
H(1,129) = 0;
H(1,257) = 1;
H(129,1) = 0;
H(129,129) = 1;
H(129,257) = 0;
H(257,1) = 1;
H(257,129) = 0;
H(257,257) = 1;

for r = 1:N+1
    A(r,r) = -1;
end
for r = 1:N
    A(r,r+1) = 1;
end

%smoothness and points, Lagrangian
A = (1/N)*A;
hx = A*H;
hy = H*A';
lambda = 1;
c = 1;
% g = H - Hp; 
M = hx'*hx + hy'*hy;
h = abs(H(1,1)-1)+abs(H(1,129))+abs(H(1,257)-1)+abs(H(129,1))+abs(H(129,129)-1)+abs(H(129,257))+abs(H(257,1)-1)+abs(H(257,129))+abs(H(257,257)-1);

fun = @(H) sum(abs(M(:))) - lambda*h + (c/2)*(h*h);

if nargout > 1 % gradient required
    g = sum(2*sparse(A'*A));  
end

[H, fval] = fminunc(fun, H);
function [f,g]

