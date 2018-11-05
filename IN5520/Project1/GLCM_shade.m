
function [glcm_shade] = GLCM_shade(glcm)

% Compute the cluster shade of GLCM

% Size of data
[M,N] = size(glcm);

% Compute the mean in x
Mean_x = GLCM_mean_x(glcm);
% Compute the mean in y
Mean_y = GLCM_mean_y(glcm);

% Define the cluster shade 
CSH   = zeros(M,N); 

% Compute the clsuter shade at each point in the matrix
for i = 1:M
    for j = 1:N 
        
        CSH(i,j) = glcm(i,j)*(i + j - Mean_x - Mean_y)^3;
        
    end
end

glcm_shade = sum(sum(CSH));