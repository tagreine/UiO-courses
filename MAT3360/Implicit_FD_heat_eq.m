%MAT-INF3360 Oblig 2 Implementation of implicit finite difference scheme

%Initial parameters

clear all,close all; clc

dt = 0.00125;
dx = 0.05;
r = dt/(dx)^2;

eps = 0.5;



%Initialization function

xmax = 1;

x = 0:dx:(xmax-dx*xmax);

f = zeros(1,length(x));

for i = 1:length(x)

        if x(i) <= 1/2
            
            f(i) = 2*x(i);
        else if x(i) > 1/2
            
            f(i) = 2*(1 - x(i));
            
            end
        end
    
end

plot(x,f)

f = f + 0.5*rand(1,length(x));

%Time vector

T = 0:dt:1;

N = length(T)-1;

n = length(f);

v = zeros(N,n);

% Initialiation and initial conditions

v(1,:) = f;

v(:,1) = 0;

v(:,end) = 0;

%Implicit scheme v^m+1 = B^-1v^m

I = eye(n,n);

toep = [2 -1 zeros(1,n-2)];

A = (1/dx^2)*toeplitz(toep);

Bin = inv(((1 - dt)*I + eps*dt*A));


for i = 1:N;
    
    v(i+1,:) = Bin*v(i,:)';
    
    v(:,1) = 0;

    v(:,end) = 0;
    
end

subplot(1,2,1)
imagesc(x,T,v)

xlabel('x')
ylabel('Time')

title('Finite difference solution to u_t = \epsilon u_{xx} + u (\epsilon = 0.05,dt=0.0025)')

subplot(1,2,2)

plot(x,v(1/dt,:))

xlabel('x')
ylabel('u(x,0.1)')

title('Time = 0.8')