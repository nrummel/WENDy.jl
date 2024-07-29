
% Use a stiff MATLAB D.E. solver, ode15s:
%
% [tout,xout]=ode15s(deRHS,[tstart,tfinal],X0,pars)
%
% here w0=[x0,y0]'', a column vector of initial values

close all;
tgrid = [0:0.1:60];
X0=[7,-10]';

pars = [72 1 2 1 1]';
[tout,xout]=ode15s(@deRHS,tgrid,X0,[],pars);

x = xout(:,1);
y = xout(:,2);
figure(1)
plot(tgrid,x,'r')
hold on
plot(tgrid,y,'b')
title('Solution')
legend('x(t)','y(t)')

% add noise
vareps = 1;
epsx=normrnd(0,sqrt(vareps),size(tout));
epsy=normrnd(0,sqrt(vareps),size(tout));
datax = x + epsx;
datay = y + epsy;
hold on
plot(tout,datax,'r*')
hold on
plot(tout,datay,'b*')

%loglikelihood = log-posterior under flat priors
dataxM=[];
datayM=[];
dataxM = datax*ones(size(tout))';
datayM = ones(size(tout))*datay';

logp1 = [];
logp1 = -log(normpdf(dataxM,x*ones(size(tout))',sqrt(vareps)))- log(normpdf(datayM,ones(size(tout))*y',sqrt(vareps)));
figure();
surf(dataxM,datayM,exp(logp1));



%%% LOOPS OVER k3 and k4:
logpx = [];
logpy = [];
logpX = [];

k3grid = .1:.1:5;
k4grid = .1:.1:5;
for i =1:length(k3grid),
    for j =1:length(k4grid),
        k3 = k3grid(i);
        k4 = k4grid(j);
        X0=[7,-10]';
        pars = [72 1 k3 k4 1]';

        [tout,xout]=ode15s(@deRHS,tgrid,X0,[],pars);
        x=xout(:,1); y=xout(:,2);
        n=length(x);

        %logpx(i,j) = sum(log(normpdf(datax,x,sqrt(vareps))));
        logpx(i,j) =  -sum((datax-x).^2)/(2*vareps) - n/2*log(2*pi*vareps);

        %logpy(i,j) = sum(log(normpdf(datay,y,sqrt(vareps))));
        logpy(i,j) =  -sum((datay-y).^2)/(2*vareps) - n/2*log(2*pi*vareps);

        logpX(i,j) = logpx(i,j)+logpy(i,j);
    end
end


figure(3)
surfl(k3grid(10:i),k4grid(10:j),(logpx(10:i,10:j))); colorbar;
figure(4)
surf(k3grid,k4grid,(logpy)); colorbar;
figure(5)
surfl(k3grid,k4grid,(logpX)); colorbar;

colormap(hsv) %rainbow colormap!
