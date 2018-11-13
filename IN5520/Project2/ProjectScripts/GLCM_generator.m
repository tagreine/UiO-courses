
function [GLCM_] = GLCM_generator(I,dx,dy,G)

% Gray level co-occurence matrix GLCM

Max = G-1;

[M,N] = size(I);

padx = abs(dx);
pady = abs(dy);  
img = double(padarray(I,[padx pady],'symmetric'));

% Computing GLCM in east and west direction with dx,dy pixel offset
GLCM_ = zeros(Max+1,Max+1);

for i= (1 + pady):(M - abs(dy))
    for j = (1 + padx):(N - abs(dx))
        
        a = img(i,j);
        b = img(i+dy,j+dx);
        
        % a + 1 and b + 1 bcs since a and b could be 0
        GLCM_(a + 1,b + 1) = GLCM_(a + 1,b + 1) + 1;
        
    end
end
