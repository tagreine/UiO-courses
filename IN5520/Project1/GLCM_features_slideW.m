
function [GLCM_features] = GLCM_features_slideW(I,W,dx,dy,direction)

% Computing the GLCM feautures like variance, contrast, inverse difference moment, cluster shade and entropy
%
% I = input image
% W = wondow size
% dx, dy = offset
% direction = textures dominant direction
% 
% GLCM_features = structure with
% 
% SpeciesName = {
%     'Var'
%     'Con'
%     'IDM'
%     'Ent'
%     'CSH'};
%
% Chose directions and merges for improving GLCM directional 
% dependence 
% a = isotropic; b = North-South; c = East-West; d = diagonal-right; 
% e = diagonal-left f = b+c; g = b+d; h = b+e; i = c+d; j = c+e; 
% k = d+e;

pad = (W - 1)/2;

% Padding image so input and output image will be same size
img   = double(padarray(I,[pad pad],'symmetric'));

% Size of data
[M,N] = size(img);

% Define variance, mean, contrast and entropy from GLCM
Variance = zeros(M,N);
Contrast = zeros(M,N);
InvDiff  = zeros(M,N);
Entropy  = zeros(M,N);
ClustSh  = zeros(M,N);



for i = (pad + 1):(M - pad)
    for j = (pad + 1):(N - pad)
        
        Window = img((i-pad):(i+pad),(j-pad):(j+pad));
   
        % Compute GLCM within the sliding windows
        
        [GLCM_H,GLCM_V,GLCM_EE,GLCM_VV] = GLCM_gen(Window,dx,dy);
        
        % Chose directions and merges for improving GLCM directional 
        % dependence 
        % a = isotropic; b = North-South; c = East-West; d = diagonal-right; 
        % e = diagonal-left f = b+c; g = b+d; h = b+e; i = c+d; j = c+e; 
        % k = d+e; 
        if direction == 'a'
           glcm = (GLCM_H+GLCM_V+GLCM_EE+GLCM_VV)/4;
        else if direction == 'b'
                glcm = GLCM_V;
        else if direction == 'c'
                glcm = GLCM_H;
        else if direction == 'd'
                glcm = GLCM_EE;
        else if direction == 'e'
                glcm = GLCM_VV;        
        else if direction == 'f'
                glcm = (GLCM_V+GLCM_H)/2;
        else if direction == 'g'
                glcm = (GLCM_V+GLCM_EE)/2;            
        else if direction == 'h'
                glcm = (GLCM_V+GLCM_VV)/2;            
        else if direction == 'i'
                glcm = (GLCM_H+GLCM_EE)/2;          
        else if direction == 'j'
                glcm = (GLCM_H+GLCM_VV)/2;
        else if direction == 'k'
                glcm = (GLCM_EE+GLCM_VV)/2;                 
            end
            end
            end
            end
            end
            end
            end
            end
            end
            end
        end
        
        % Compute the different features
        
        Variance(i,j) = GLCM_var(glcm,'x');
        Contrast(i,j) = GLCM_con(glcm);
        InvDiff(i,j)  = GLCM_IDM(glcm);
        Entropy(i,j)  = GLCM_ent(glcm);
        ClustSh(i,j)  = GLCM_shade(glcm);
    
    end
end

Var  = Variance((pad + 1):(end - pad),(pad + 1):(end - pad));
Con  = Contrast((pad + 1):(end - pad),(pad + 1):(end - pad));
IDM  = InvDiff((pad + 1):(end - pad),(pad + 1):(end - pad));
Ent  = Entropy((pad + 1):(end - pad),(pad + 1):(end - pad)); 
CSH  = ClustSh((pad + 1):(end - pad),(pad + 1):(end - pad)); 

SpeciesName = {
    'Var'
    'Con'
    'IDM'
    'Ent'
    'CSH'};

C = SpeciesName';
C(2,:) = num2cell(zeros(size(C)));
C(2,1) = {Var};
C(2,2) = {Con};
C(2,3) = {IDM};
C(2,4) = {Ent};
C(2,5) = {CSH};
GLCM_features = struct(C{:});
    



