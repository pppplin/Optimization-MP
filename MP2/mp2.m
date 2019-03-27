N = 256;
vh = zeros(N+1);

vh(1,1) = 1;
vh(1,129) = 0;
vh(1,257) = 1;
vh(129,1) = 0;
vh(129,129) = 1;
vh(129,257) = 0;
vh(257,1) = 1;
vh(257,129) = 0;
vh(257,257) = 1;

%smoothness and points, Lagrangian
hstr = diag([1,repelem(2, (N+1)^2-1)]);
for i=1:(N+1)^2-1
    hstr(i,i+1)=-1;
    hstr(i+1,i)=-1;
end
options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessPattern',hstr);
%options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn', 'objective'); 
%options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true); 
% v should be the final h(x,y) (height) matrix. 
[v, FVAL] = fminunc(@smoothnessAL, vh, options);

% plot 
[X,Y] = meshgrid(0:1/N:1);
figure
mesh(X,Y,v)

% problem now. Not sure if hessian is correct, and sparse???
