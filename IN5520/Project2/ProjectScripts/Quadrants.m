
function [Q1,Q2,Q3,Q4] = Quadrants(I)

% Divide the image into 4 quadrants

[M,N] = size(I);

G = M;
     
Q1 = I(1:G/2,1:G/2);
Q2 = I(1:G/2,G/2+1:G);
Q3 = I(G/2+1:G,1:G/2);
Q4 = I(G/2+1:G,G/2+1:G);

end