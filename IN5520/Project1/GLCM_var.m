
function [glcm_var] = GLCM_var(glcm,dir)

% Compute the variance of GLCM

% Size of data
[M,N] = size(glcm);

% Compute the mean
if dir == 'x'
   Mean = GLCM_mean_x(glcm);
else if dir == 'y'
   Mean = GLCM_mean_y(glcm);
    end
end

% Define the variance 
Var   = zeros(M,N); 

% Compute the variance at each point in the matrix
for i = 1:M
    for j = 1:N 
        
        Var(i,j) = glcm(i,j)*(i - Mean)^2;
        
    end
end

glcm_var = sum(sum(Var));