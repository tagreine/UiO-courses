
function [glcm_IDM] = GLCM_IDM(glcm)

% Compute the Inverse Difference Moment (IDM) of GLCM

% Size of data
[M,N] = size(glcm);

% Define the IDM
IDM   = zeros(M,N); 

% Compute the IDM at each point in the matrix
for i = 1:M
    for j = 1:N 
        
        A = 1/(1 + (i - j)^2);
        IDM(i,j) = glcm(i,j)*A;
        
    end
end

glcm_IDM = sum(sum(IDM));