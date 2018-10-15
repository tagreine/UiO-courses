function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% =============================================================

z     = X*theta;
h     = sigmoid(z);

J        = (1/m)*(-y'*log(h) - (1 - y)'*log(1 - h)) + lambda/(2*m)*(theta(2:end)')*theta(2:end);
grad0    = (1/m)*X(:,1)'*(h - y);
gradReg  = (1/m)*X(:,2:end)'*(h - y) + (lambda/m)*theta(2:end);
grad     = [grad0;gradReg];

% =============================================================

end