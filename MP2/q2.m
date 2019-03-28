% A = [0 0; 0 0.5; 0 1; 0.5 0; 0.5 0.5; 0.5 1; 1 0; 1 0.5; 1 1];
% B = [1;0;1;0;1;0;1;0;1];
% x = linsolve(A, B); 


[X, Y] = meshgrid(0:0.5:1);
V = [1 0 1; 0 1 0; 1 0 1];
[Xq, Yq] = meshgrid(0:0.01:1);
Vq = interp2(X,Y,V,Xq,Yq);
mesh(Xq, Yq, Vq)