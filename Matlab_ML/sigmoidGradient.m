function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z.

% =============================================================

g1     = 1./(1 + exp(-z));
g    = g1.*(1 - g1);

% =============================================================

end