
function [FeatureQ1,FeatureQ2,FeatureQ3,FeatureQ4] = GLCM_subFeature(I,W,dx,dy,G)

pad = (W - 1)/2;

% Padding image so input and output image will be same size
img   = double(padarray(I,[pad pad],'symmetric'));

% Size of data
[M,N] = size(img);

FeatureQ1 = zeros(M,N);
FeatureQ2 = zeros(M,N);
FeatureQ3 = zeros(M,N);
FeatureQ4 = zeros(M,N);

for i = (pad + 1):(M - pad)
    for j = (pad + 1):(N - pad)
        
        Window = img((i-pad):(i+pad),(j-pad):(j+pad));
        
        Window_GLCM = GLCM_generator(Window,dx,dy,G);
        
        [Q1,Q2,Q3,Q4]  = Quadrants(Window_GLCM);
        FeatureQ       = featureQ(Q1,Q2,Q3,Q4,Window_GLCM);
        FeatureQ1(i,j) = FeatureQ(1,1);
        FeatureQ2(i,j) = FeatureQ(1,2);
        FeatureQ3(i,j) = FeatureQ(2,1);
        FeatureQ4(i,j) = FeatureQ(2,2);
        
    end
end

FeatureQ1 = FeatureQ1((pad + 1):(end - pad),(pad + 1):(end - pad));
FeatureQ2 = FeatureQ2((pad + 1):(end - pad),(pad + 1):(end - pad));
FeatureQ3 = FeatureQ3((pad + 1):(end - pad),(pad + 1):(end - pad));
FeatureQ4 = FeatureQ4((pad + 1):(end - pad),(pad + 1):(end - pad));
end