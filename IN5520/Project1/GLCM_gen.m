

function [GLCM_H,GLCM_V,GLCM_EE,GLCM_VV] = GLCM_gen(I,dx,dy)

% Gray level co-occurence matrix GLCM

Max = max(max(I));

[M,N] = size(I);

% Computing GLCM in east and west direction with dx,dy pixel offset
GLCM_E = zeros(Max+1,Max+1);

for i= 1:M
    for j = 1:N-dx
        
        a = I(i,j);
        b = I(i,j + dx);
        
        % a + 1 and b + 1 bcs since a and b could be 0
        GLCM_E(a + 1,b + 1) = GLCM_E(a + 1,b + 1) + 1;
        
    end
end

GLCM_W  = GLCM_E';

% make the horizontal glcm symmetrical and normalized
pixpair1 = sum(sum(GLCM_E + GLCM_W)); 
GLCM_H = (GLCM_E + GLCM_W)/pixpair1;

% Computing GLCM in east and west direction with 1 pixel offset
GLCM_S = zeros(Max+1,Max+1);

for i= 1:M-dy
    for j = 1:N
        
        a = I(i,j);
        b = I(i + dy,j);
        
        % a + 1 and b + 1 bcs since a and b could be 0
        GLCM_S(a + 1,b + 1) = GLCM_S(a + 1,b + 1) + 1;
        
    end
end

GLCM_N = GLCM_S';

% make the vertical glcm symmetrical and normalized
pixpair2 = sum(sum(GLCM_N + GLCM_S));
GLCM_V = (GLCM_N + GLCM_S)/pixpair2;

% Computing GLCM diagonal-right direction with 1 pixel offset (corresponds to 135 and 315 degree)
GLCM_E2 = zeros(Max+1,Max+1);

for i= 1:M-dy
    for j = 1:N-dx
        
        a = I(i,j);
        b = I(i + dy,j + dx);
        
        % a + 1 and b + 1 bcs since a and b could be 0
        GLCM_E2(a + 1,b + 1) = GLCM_E2(a + 1,b + 1) + 1;
        
    end
end

GLCM_E3 = GLCM_E2';

% make the vertical glcm symmetrical and normalized
pixpair3 = sum(sum(GLCM_E2 + GLCM_E3));
GLCM_EE = (GLCM_E2 + GLCM_E3)/pixpair3;

% Computing GLCM diagonal-left direction with 1 pixel offset (corresponds to 45 and 225 degree)
GLCM_V2 = zeros(Max+1,Max+1);

for i= 1:M-dy
    for j = dx+1:N
        
        a = I(i,j);
        b = I(i + dy,j - dx);
        
        % a + 1 and b + 1 bcs since a and b could be 0
        GLCM_V2(a + 1,b + 1) = GLCM_V2(a + 1,b + 1) + 1;
        
    end
end

GLCM_V3 = GLCM_V2';

% make the vertical glcm symmetrical and normalized
pixpair4 = sum(sum(GLCM_V2 + GLCM_V3));
GLCM_VV = (GLCM_V2 + GLCM_V3)/pixpair4;








   
        