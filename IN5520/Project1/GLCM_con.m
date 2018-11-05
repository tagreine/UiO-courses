
function [glcm_con] = GLCM_con(glcm)

% Compute the contrast of GLCM

[M,N] = size(glcm);

% Define the contrast
con   = zeros(M,N); 

% Compute constrast at every point in the glcm matrix
for i = 1:M
    for j = 1:N 
        
        con(i,j) = glcm(i,j)*(i - j)^2;
        
    end
end

glcm_con = sum(sum(con));
