function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% =========================================================================

p_tmp = zeros(m,1);
h = zeros(m,num_labels); 

for i = 1:num_labels
    h(:,i) = sigmoid(X*all_theta(i,:)');
end

%h = max(h,[],2);

for j = 1:m
    [a,b] = max(h(j,:));
    
    if a >= 0.5
        
        p_tmp(j) = b;
    else
        
        p_tmp(j) = 0;

    end
end

p = p_tmp;

% =========================================================================

end