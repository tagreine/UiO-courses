
function [glcm_ent] = GLCM_ent(glcm)

% Compute the GLCM entropy

[M,N] = size(glcm);

ent   = zeros(M,N); 

for i = 1:M
    for j = 1:N 
        
        if glcm(i,j) == 0 
            ent(i,j) = 0;
        else
            ent(i,j) = glcm(i,j)*(-log(glcm(i,j)));
        end
    end
end

glcm_ent = sum(sum(ent));