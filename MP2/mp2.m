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
%options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true); 
%optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true); 
options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true); 
[v, fval] = fminunc(@smoothnessAL, vh);


