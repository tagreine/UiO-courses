
function [class_means,CoVar] = MultiGaussTraining(features,training_mask,labels) 

num_features = numel(features);
num_labels = numel(labels);
% Assign class and feature means
class_means  = zeros(num_features,num_labels);

% Training mean
for ii = 1:num_features
    tmp_data = features{ii};
    for jj = 1:num_labels
        class_tmp_data = tmp_data(training_mask == jj);      
        class_means(ii,jj) = mean(class_tmp_data); 
    end
end

% Training covariance
% Compute the covariance between the features
CoVar        = zeros(num_features, num_features, num_labels);
% Training covariance
for jj = 1:num_labels
    tmp          = features{1};
    tmp2         = tmp(training_mask == jj);
    mask_len     = length(tmp2);  
    tmp_features = zeros(mask_len,num_features);
    
    for ii = 1:num_features
        tmp_data           = features{ii};
        tmp_data2          = tmp_data(training_mask == jj);
        tmp_features(:,ii) = tmp_data2;   
    end
    
    CoVar(:,:,jj) = cov(tmp_features);   
end




