function Xdot = deRHS(t,X,pars)

x = X(1);
y = X(2);
%pars = [72 1 2 1 1]';

xdot = pars(1)/(36+pars(2)*y)-pars(3); 
ydot = pars(4)*x - pars(5);

Xdot = [xdot,ydot]';
