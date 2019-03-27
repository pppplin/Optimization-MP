function stop = myplot(x,optimValues,state)
N = 10;
% plot 
[X,Y] = meshgrid(0:1/N:1);
%figure
mesh(X,Y,x)
stop = false;