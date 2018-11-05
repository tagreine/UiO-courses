

function [glcm_mean] = GLCM_mean_x(glcm)

% Compute the GLCM mean

% Size of data
[M,N] = size(glcm);

% Define the mean 
Mean   = zeros(M,N); 

% Compute the mean at each point in the matrix for x direction
for i = 1:M
    for j = 1:N 
         
        Mean(i,j) = glcm(i,j)*i;
        
    end
end

glcm_mean = sum(sum(Mean));