function diff_im = anisodiff2D_dir(im, num_iter, delta_t, kappa, sigma, option)
%
%   
%
% 
%
%
%
%
%
%
%
% Convert input image to double.
im = double(im);
% Extracting size of input model
[M N] = size(im);

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

%------------------------------------------------------------------
% Applying orientation dependent smoothing
% Compute the gradients
% [X Y] = gradient(diff_im);
% theta = atan(Y./X)+pi/2;
%  
% theta_new = zeros(M,N);
%  
% % Compute the orientation dependent weights
%  
% for i = 1:M
%     for j = 1:N
%         if theta(i,j) >= pi/2
%            theta_new(i,j) = pi-theta(i,j);
%         else if theta(i,j) < pi/2
%            theta_new(i,j) = theta(i,j);
%              end
%         end
%     end
% end
%------------------------------------------------------------------

%stencil2D = [0 0 0;0 1 0;0 0 0];
%stencil2D = [0 0.15 0;0.15 04 0.15;0 0.15 0];
stencil2D   = GausFunc(9,sigma);
%stencil2D_0 = GausFunc(15,2);

% Anisotropic diffusion.
for t = 1:num_iter
        
    
        diff_im2 = conv2(diff_im,stencil2D,'same');
        
        [X Y] = gradient(diff_im2);
        theta = atan(Y./X)+pi/2;
        %theta = conv2(theta,stencil2D,'same');
 
        theta_new = zeros(M,N);
%       theta_NS = zeros(M,N);
%       theta_EW = zeros(M,N);
 
        % Compute the orientation dependent weights
 
         for i = 1:M
             for j = 1:N
                 if theta(i,j) > pi/2
                     theta_new(i,j) = pi-theta(i,j);
                 else if theta(i,j) <= pi/2
                     theta_new(i,j) = theta(i,j);
                     end
                 end
             end
         end
%         
%         for i = 1:M
%             for j = 1:N
%                 if theta_new(i,j) > pi/4
%                     theta_NS(i,j) = theta_new(i,j);
%                 else if theta(i,j) <= pi/4
%                     theta_NS(i,j) = 0;
%                     end
%                 end
%             end
%         end
%         for i = 1:M
%             for j = 1:N
%                 if theta_new(i,j) > pi/4
%                     theta_EW(i,j) = 0;
%                 else if theta(i,j) <= pi/4
%                     theta_EW(i,j) = theta_new(i,j);
%                     end
%                 end
%             end
%         end

         
        % Finite differences. [imfilter(.,.,'conv') can be replaced by conv2(.,.,'same')]
        nablaN  = conv2(diff_im2,hN,'same');
        nablaS  = conv2(diff_im2,hS,'same');   
        nablaW  = conv2(diff_im2,hW,'same');
        nablaE  = conv2(diff_im2,hE,'same');   
        nablaNE = conv2(diff_im2,hNE,'same');
        nablaSE = conv2(diff_im2,hSE,'same');   
        nablaSW = conv2(diff_im2,hSW,'same');
        nablaNW = conv2(diff_im2,hNW,'same'); 
        
        % Regularization
        %stencil2D = [0 0.1 0;0.1 0.6 0.1;0 0.1 0];
        %nablaN2  = conv2(nablaN,stencil2D,'same');
        %nablaS2  = conv2(nablaS,stencil2D,'same');   
        %nablaW2  = conv2(nablaW,stencil2D,'same');
        %nablaE2  = conv2(nablaE,stencil2D,'same');   
        %nablaNE2 = conv2(nablaNE,stencil2D,'same');
        %nablaSE2 = conv2(nablaSE,stencil2D,'same');   
        %nablaSW2 = conv2(nablaSW,stencil2D,'same');
        %nablaNW2 = conv2(nablaNW,stencil2D,'same');         
        
        
        
        % Diffusion function.
        if option == 1
            cN = exp(-(abs(nablaN)/kappa).^2);
            cS = exp(-(abs(nablaS)/kappa).^2);
            cW = exp(-(abs(nablaW)/kappa).^2);
            cE = exp(-(abs(nablaE)/kappa).^2);
            cNE = exp(-(abs(nablaNE)/kappa).^2);
            cSE = exp(-(abs(nablaSE)/kappa).^2);
            cSW = exp(-(abs(nablaSW)/kappa).^2);
            cNW = exp(-(abs(nablaNW)/kappa).^2);
        elseif option == 2
            cN = 1./(1 + (abs(nablaN)/kappa).^2);
            cS = 1./(1 + (abs(nablaS)/kappa).^2);
            cW = 1./(1 + (abs(nablaW)/kappa).^2);
            cE = 1./(1 + (abs(nablaE)/kappa).^2);
            cNE = 1./(1 + (abs(nablaNE)/kappa).^2);
            cSE = 1./(1 + (abs(nablaSE)/kappa).^2);
            cSW = 1./(1 + (abs(nablaSW)/kappa).^2);
            cNW = 1./(1 + (abs(nablaNW)/kappa).^2);
        end

        % Applying boundary conditions to the diffusion function
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
        
        
      
               diff_im = diff_im + ...
                         delta_t*(...
                         (1/(dy^2))*cN.*nablaN.*sin(theta_new) + (1/(dy^2))*cS.*nablaS.*sin(theta_new) + ...
                         (1/(dx^2))*cW.*nablaW.*cos(theta_new) + (1/(dx^2))*cE.*nablaE.*cos(theta_new) + ...
                         (1/(dd^2))*cNE.*nablaNE + (1/(dd^2))*cSE.*nablaSE + ...
                         (1/(dd^2))*cSW.*nablaSW + (1/(dd^2))*cNW.*nablaNW );

%               diff_im = diff_im + ...
%                         delta_t*(...
%                         (1/(dy^2))*cN.*nablaN.*sin(theta_NS) + (1/(dy^2))*cS.*nablaS.*sin(theta_NS) + ...
%                         (1/(dx^2))*cW.*nablaW.*cos(theta_EW) + (1/(dx^2))*cE.*nablaE.*cos(theta_EW) + ...
%                         (1/(dd^2))*cNE.*nablaNE.*sin(theta) + (1/(dd^2))*cSE.*nablaSE.*sin(theta) + ...
%                         (1/(dd^2))*cSW.*nablaSW.sin(theta) + (1/(dd^2))*cNW.*nablaNW.*sin(theta) );

                

           
        % Iteration warning.
        fprintf('\rIteration %d\n',t);
end