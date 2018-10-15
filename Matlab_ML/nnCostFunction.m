function [J,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% recoding y
eye_matrix = eye(num_labels);
y = eye_matrix(y,:);

% =========================================================================

% Forward propagation

a1     = [ones(m,1),X];
z2     = a1*Theta1';
a2     = [ones(m,1) sigmoid(z2)];
z3     = a2*Theta2';
h      = sigmoid(z3);

% =========================================================================

% Cost function

for i = 1:num_labels

    J1(i) = (1/m)*(-y(:,i)'*log(h(:,i)) - (1 - y(:,i))'*log(1 - h(:,i)));

end

% RegularizationS

for i = 1:hidden_layer_size
   
    J2(i) = lambda/(2*m)*(Theta1(i,2:end)*Theta1(i,2:end)') ; 
    
end

for i = 1:num_labels
   
    J3(i) =  lambda/(2*m)*(Theta2(i,2:end)*Theta2(i,2:end)');
    
end


J = sum(J1) + sum(J2) + sum(J3);

% =========================================================================

% Backward propagation with regularization

delta3 = zeros(m,num_labels);

for k = 1:m
   delta3(k,:) = h(k,:) - y(k,:);
end

G = sigmoidGradient(z2);

delta2 = delta3*Theta2(:,2:end).*G;

Delta11 = (delta2')*a1(:,1);
Delta12 = (delta2')*a1(:,2:end) + lambda*Theta1(:,2:end);

Delta1 = (1/m)*[Delta11 Delta12];

Delta21 = (delta3')*a2(:,1);
Delta22 = (delta3')*a2(:,2:end) + lambda*Theta2(:,2:end);

Delta2 = (1/m)*[Delta21 Delta22];

Theta1_grad = Delta1;
Theta2_grad = Delta2;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end