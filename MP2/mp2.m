N = 256;
H = zeros(N+1, N+1);

H(1,1) = 1;
H(1,129) = 0;
H(1,257) = 1;
H(129,1) = 0;
H(129,129) = 1;
H(129,257) = 0;
H(257,1) = 1;
H(257,129) = 0;
H(257,257) = 1;

%smoothness and points, Lagrangian
%options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true);
[H, fval] = fminunc(@smoothnessAL, H);


