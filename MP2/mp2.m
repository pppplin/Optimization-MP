N = 256;
M = N + 1;
vh = zeros(M);

half = ceil( M / 2 );

vh(1,1) = 1;
vh(1,half) = 0;
vh(1,M) = 1;
vh(half,1) = 0;
vh(half,half) = 1;
vh(M,M) = 0;
vh(M,1) = 1;
vh(M,half) = 0;
vh(M,M) = 1;

D = sparse(1:M*M,1:M*M,ones(1,M*M),M*M,M*M);
E = sparse(M + 1:M*M,1:M*M-M,ones(1,M*M-M),M*M,M*M);
F = sparse(2:M*M,1:M*M-1,ones(1,M*M-1),M*M,M*M);
hstr = E+D+E'+F+F';
options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessPattern',hstr,'MaxIterations', 1000, 'PlotFcn', @myplot);
%options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn', 'objective'); 
%options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true); 
% v should be the final h(x,y) (height) matrix. 
[v, FVAL] = fminunc(@smoothnessAL, vh, options);



% problem now. Not sure if hessian is correct, and sparse???
