
function [pred_class] = MultGaussClassificationSVD(features,labels,class_means,CoVar)

[M,N] = size(features{1});

% Assign the predicted classes
pred_class = zeros(M,N);
% Assign number of features and number of labels
num_features = numel(features);
num_labels   = numel(labels);

for r = 1:M
    for c = 1:N
        pixel_val = zeros(num_features,1);
        
        % Extract the pixel value for all features
        for i = 1:num_features
            tmp_data     = features{i};
            val          = tmp_data(r,c);
            pixel_val(i) = val;
        end
        
        % Assign predicted classes
        max_label = 0;
        max_val   = 0;
        
        % Predict which class the extracted pixel position belongs to
        for i = 1:num_labels
            % Extracte mean, covariance and class from all labels
            class = labels(i);
            mu    = class_means(:,i);
            CV    = CoVar(:,:,i);
            Gauss = MultiGaussSVD(pixel_val,CV,mu,num_labels,num_features);
            if i == 1
                max_label = class;
                max_val   = Gauss;
            elseif Gauss > max_val
                max_label = class;
                max_val   = Gauss;            
            end
        end
        % Assign the predicted class
        pred_class(r, c) = max_label;
    end
end
end
