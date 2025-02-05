function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve.

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% =========================================================================

for i = 1:m
    % Compute train/cross validation errors using training examples 
    theta           = trainLinearReg(X(1:i,:), y(1:i), lambda);
    error_train(i)  = linearRegCostFunction(X(1:i,:), y(1:i), theta, 0);
    error_val(i)    = linearRegCostFunction(Xval, yval, theta, 0);         
end


% =========================================================================

end