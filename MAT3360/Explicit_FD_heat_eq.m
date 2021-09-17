%MAT-INF3360 Oblig 2 Implementation of explicit finite difference scheme

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

%Time vector

T = 0:dt:1;

N = length(T)-1;

n = length(f);

v = zeros(N,n);

% Initialiation and boundary conditions

v(1,:) = f;

v(:,1) = 0;

v(:,end) = 0;


%scheme 
%v(i+1,j) = eps*dt/dx^2*v(i,j-1) + eps*dt/dx^2*v(i,j+1) + (1 + dt - 2*eps*dt/dx^2)*v(i,j) 

C1 = eps*r;
C2 = (1 + dt - 2*eps*r);

for i = 1:N
    
    for j = 2:n-1
        
    v(i+1,j) = C1*v(i,j-1) + C1*v(i,j+1) + C2*v(i,j); 
    
    end
end

subplot(1,2,1)
imagesc(x,T,v)

xlabel('x')
ylabel('Time')

title('Finite difference solution to u_t = \epsilon u_{xx} + u (\epsilon =0.5, dt=0.00125)')

subplot(1,2,2)

plot(v(1/dt,:))

xlabel('x')
ylabel('u(x,0.1)')

title('Time = 0.1')


