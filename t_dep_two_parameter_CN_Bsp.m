% t_dep_two_parameter_CN_Bsp.m
% This script solves a time-dependent convection-diffusion equation using
% the Crank-Nicolson method with trigonometric quintic B-spline collocation.
% It is designed for a problem with two parameters.

% If you are using this code, please cite our paper:
% "A study on Time-DependentTwo Parameter Singularly Perturbed Problems via
% Trigonometric Quintic B-splines on an Exponentially Graded mesh"

clear; clc;
WarnState = warning('off');
format long

% --- PART 1: Parameter and Grid Setup ---

Problem  = 5;           % 1 For problem 1 and 2 for Problem 2
eps = 10e-2; mu = 1; 
M = 32; N = 10;         % M and N are number of space and time points, resp.
theta = 0.5;              % Theta prameter 0.5 for CN,

%==========================================================================
if Problem ==1
    %%%%%%%% Parameters of the problem 1 %%%%%%%%%%
    a = @(x,t) (1+x); da = @(x,t) 1;
    b = @(x,t) 1;     db = @(x,t) 0;
    c = @(x) b(x,0);  
    f = @(x,t) -16*x.^2.*(1-x).^2;
    df = @(x,t) -32*x.*(2*x.^2-3*x+1);
    u0 = @(x) zeros(size(x));
    tmin = 0; tmax = 1; 
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif Problem ==2
    %%%%%%%% Parameters of the problem 2 %%%%%%%%%%
    a = @(x,t) (1+x-x.^2+t.^2); da = @(x,t) (1-2*x+t.^2);
    b = @(x,t) (1+5*x.*t);     db = @(x,t) 5*t;
    c = @(x) b(x,0);  
    f = @(x,t) (x-x.^2).*(1 - exp(t));
    df = @(x,t) (1-2*x).*(1-exp(t));
    u0 = @(x) zeros(size(x));
    tmin = 0; tmax = 1; 
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif Problem ==3
    %%%%%%%% Parameters of the problem 3 %%%%%%%%%%
    a = @(x,t) -(1+x.^2+0.5*sin(pi*x)); da = @(x,t) -(2*x+0.5*pi*cos(pi*x));
    b = @(x,t) (1+x.^2+0.5*sin(pi*t/2));     db = @(x,t) 2*x;
    c = @(x) b(x,0);  
    f = @(x,t) x.^3.*(1-x).^3.*t.*sin(pi*t);
    df = @(x,t) -3*t.*((1-x).^2).*x.^2.*(2*x-1).*sin(pi*t);
    u0 = @(x) zeros(size(x));
    tmin = 0; tmax = 1; 
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 elseif Problem ==4
    %%%%%%%% Parameters of the problem 4 %%%%%%%%%%
    a = @(x,t) (1+exp(x)); da = @(x,t) exp(x);
    b = @(x,t) (1+x.^5);     db = @(x,t) 5*x.^4;
    c = @(x) b(x,0);  
    f = @(x,t) -10*exp(t.^2).*x.^2.*(1-x.^2);
    df = @(x,t) 20*exp(t.^2).*x.*(2*x.^2-1);
    u0 = @(x) zeros(size(x));
    tmin = 0; tmax = 1; 
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif Problem == 5
    %%%%%%%% Parameters of the problem 5 %%%%%%%%%%
    a = @(x,t) -1; da = @(x,t) 0;
    b = @(x,t) 0; db = @(x,t) 0;
    c = @(x) 1;
    f = @(x,t) exp(-t).*(-exp(-1/eps)+(1-exp(-1/eps)).*(1-x)+exp((-1/eps)*(1-x)));
    df = @(x,t) exp(-t).*((exp(-1/eps)-1)+exp((-1/eps)*(1-x))*(1/eps));
    u0 = @(x) exp(-1/eps)+(1-exp(-1/eps)).*x-exp(-1/eps*(1-x));
    tmin = 0; tmax = 2; 
    u_exact = @(x,t) exp(-t).*(exp(-1/eps) + (1-exp(-1/eps)).*x - exp(-(1-x)/eps));
end

    lam0 = @(x) (mu*a(x,0)+sqrt(mu^2*a(x,0).^2+4*eps*c(x)))./(-2*eps);
    lam1 = @(x) (mu*a(x,0)-sqrt(mu^2*a(x,0).^2+4*eps*c(x)))./(-2*eps);

    v0 = -1*max(lam0(0:0.00000001:1)); v1 = min(lam1(0:0.00000001:1));
    tau = 1.1; l = 0.1;
    phi_0 = @(x) -log(1-4*(1-exp(-l*v0./(2*tau))).*x);
    phi_1 = @(x) -log(1-4*(1-exp(-l*v1./(2*tau))).*(1-x));

    if mod(M,4) ~= 0
        fprintf('Error: M must need to be devisible of 4\n');
        return
    end
    l1 = 1:M/4; l2 = M/4+1:3*M/4; l3 = 3*M/4+1:M;
    x1 = tau/(l*v0)*phi_0((l1-1)./(M-1));
    x3 = 1-tau/(l*v1)*phi_1((l3-1)./(M-1));
    x2 = x1(end) + (x3(1)-x1(end))/(M/2+1) * (l2-M/4);
%     Space and time discretization
    
    x11 = linspace(x1(1),x1(end),length(x1));
    x13 = linspace(x3(1),x3(end),length(x3));

%     x = [x11, x2, x13]; % Shishkin-type meshes6
    x = [x1, x2,x3]; % exponentially graded mesh
%     x = linspace(0,1,M); % Uniform

t = linspace(tmin,tmax,N);
dt = abs(t(2)-t(1));

% --- PART 2: B-spline Basis Matrix Calculation ---

T = getTQBSatNodes(x);
DT = getDTQBSatNodes(x);
DDT = getDDTQBSatNodes(x);
DDDT = getDDDTQBSatNodes(x);

p = ones(1,M+4);
S = zeros(N,M);
S(1,:) = u0(x);

u = zeros(M+4,1);
if Problem==5
    u = pinv(T)*u0(x)';
end

% --- PART 3: Time-Stepping Loop (Crank-Nicolson) ---

for n =2:N
    q = eye(M+4); q = q(:,3:M+2);
    A1 = q*(theta*dt*((-eps*DDT) - (kron(mu*a(x',t(n)),p).*DT) +...
        (kron(b(x',t(n)),p).*T)) + T);
    b1 = q*(dt*f(x',t(n-1)+theta*dt)+ ...
        (T-(1-theta)*dt*((-eps*DDT) - (kron(mu*a(x',t(n)),p).*DT) +...
        (kron(b(x',t(n)),p).*T)))*u);
    
    q = 0*q; q([1, end]) = 1;
    A2 = (q*T);
    b2 = q*S(n,:)';
    
    q = 0*q; q([2, end-1]) = 1;
    A3 = q*((eps*DDDT) + (kron(mu*a(x',t(n)),p).*DDT) +...
        (kron(mu*da(x',t(n))-b(x',t(n))-1/dt,p).*DT) - ...
        (kron(db(x',t(n)),p).*T));
    b3 = q*(df(x',t(n))+DT*u/dt);
    
    % --- Construction of the A Matrix (LHS) and RHS Vector ---
    A = A1 + A2 + A3;
    bf = b1+b2+b3;
    
    % Solve the linear system and 
    u = A\bf;
    
    % Update the solution 
    S(n,:) = T*u;
end
%==========================================================================

% --- PART 4: Plot the solution to visualize the result ---
surf(x,t,S,'EdgeColor','none')





