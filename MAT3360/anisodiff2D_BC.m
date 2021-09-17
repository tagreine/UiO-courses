function diff_im = anisodiff2D_BC(im, num_iter, delta_t, kappa, option)
%ANISODIFF2D Conventional anisotropic diffusion
%   DIFF_IM = ANISODIFF2D(IM, NUM_ITER, DELTA_T, KAPPA, OPTION) perfoms 
%   conventional anisotropic diffusion (Perona & Malik) upon a gray scale
%   image. A 2D network structure of 8 neighboring nodes is considered for 
%   diffusion conduction.
% 
%       ARGUMENT DESCRIPTION:
%               IM       - gray scale image (MxN).
%               NUM_ITER - number of iterations. 
%               DELTA_T  - integration constant (0 <= delta_t <= 1/7).
%                          Usually, due to numerical stability this 
%                          parameter is set to its maximum value.
%               KAPPA    - gradient modulus threshold that controls the conduction.
%               OPTION   - conduction coefficient functions proposed by Perona & Malik:
%                          1 - c(x,y,t) = exp(-(nablaI/kappa).^2),
%                              privileges high-contrast edges over low-contrast ones. 
%                          2 - c(x,y,t) = 1./(1 + (nablaI/kappa).^2),
%                              privileges wide regions over smaller ones. 
% 
%       OUTPUT DESCRIPTION:
%                DIFF_IM - (diffused) image with the largest scale-space parameter.
% 
% Version of Daniel Simoes Lopes May 2007 original version.
% Included boundary condition on diffusion function (c(outer boundary)=0)
% and changed to conv2 which makes the process much faster.

% Convert input image to double.
im = double(im);

% PDE (partial differential equation) initial condition.
diff_im = im;

% Center pixel distances.
dx = 1;
dy = 1;
dd = sqrt(2);

% 2D convolution masks - finite differences.
hN = [0 1 0; 0 -1 0; 0 0 0];
hS = [0 0 0; 0 -1 0; 0 1 0];
hE = [0 0 0; 0 -1 1; 0 0 0];
hW = [0 0 0; 1 -1 0; 0 0 0];
hNE = [0 0 1; 0 -1 0; 0 0 0];
hSE = [0 0 0; 0 -1 0; 0 0 1];
hSW = [0 0 0; 0 -1 0; 1 0 0];
hNW = [1 0 0; 0 -1 0; 0 0 0];

% Anisotropic diffusion.
for t = 1:num_iter

        % Finite differences.
        nablaN = conv2(diff_im,hN,'same');
        nablaS = conv2(diff_im,hS,'same');   
        nablaW = conv2(diff_im,hW,'same');
        nablaE = conv2(diff_im,hE,'same');   
        nablaNE = conv2(diff_im,hNE,'same');
        nablaSE = conv2(diff_im,hSE,'same');   
        nablaSW = conv2(diff_im,hSW,'same');
        nablaNW = conv2(diff_im,hNW,'same'); 
        
        % Diffusion function.
        if option == 1
            cN = exp(-(nablaN/kappa).^2);
            cS = exp(-(nablaS/kappa).^2);
            cW = exp(-(nablaW/kappa).^2);
            cE = exp(-(nablaE/kappa).^2);
            cNE = exp(-(nablaNE/kappa).^2);
            cSE = exp(-(nablaSE/kappa).^2);
            cSW = exp(-(nablaSW/kappa).^2);
            cNW = exp(-(nablaNW/kappa).^2);
        elseif option == 2
            cN = 1./(1 + (nablaN/kappa).^2);
            cS = 1./(1 + (nablaS/kappa).^2);
            cW = 1./(1 + (nablaW/kappa).^2);
            cE = 1./(1 + (nablaE/kappa).^2);
            cNE = 1./(1 + (nablaNE/kappa).^2);
            cSE = 1./(1 + (nablaSE/kappa).^2);
            cSW = 1./(1 + (nablaSW/kappa).^2);
            cNW = 1./(1 + (nablaNW/kappa).^2);
        end

        
        % Applying boundary conditions on the diffusion function
%       cN(:,1)     = 0;
%       cN(:,end)   = 0;
        cN(1,:)     = 0;
        cN(end,:)   = 0;
        
%       cS(:,1)     = 0;
%       cS(:,end)   = 0;
        cS(1,:)     = 0;
        cS(end,:)   = 0;
        
        cW(:,1)     = 0;
        cW(:,end)   = 0;
%       cW(1,:)     = 0;
%       cW(end,:)   = 0;
        
        cE(:,1)     = 0;
        cE(:,end)   = 0;
%       cE(1,:)     = 0;
%       cE(end,:)   = 0;
       
        cNE(:,1)     = 0;
        cNE(:,end)   = 0;
        cNE(1,:)     = 0;
        cNE(end,:)   = 0;
        
        cSE(:,1)     = 0;
        cSE(:,end)   = 0;
        cSE(1,:)     = 0;
        cSE(end,:)   = 0;
        
        cSW(:,1)     = 0;
        cSW(:,end)   = 0;
        cSW(1,:)     = 0;
        cSW(end,:)   = 0;
        
        cNW(:,1)     = 0;
        cNW(:,end)   = 0;
        cNW(1,:)     = 0;
        cNW(end,:)   = 0;        
            
        % Discrete PDE solution.
        diff_im = diff_im + ...
                  delta_t*(...
                  (1/(dy^2))*cN.*nablaN + (1/(dy^2))*cS.*nablaS + ...
                  (1/(dx^2))*cW.*nablaW + (1/(dx^2))*cE.*nablaE + ...
                  (1/(dd^2))*cNE.*nablaNE + (1/(dd^2))*cSE.*nablaSE + ...
                  (1/(dd^2))*cSW.*nablaSW + (1/(dd^2))*cNW.*nablaNW );
           
        % Iteration warning.
        fprintf('\rIteration %d\n',t);
end