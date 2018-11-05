
function [Variance] = variance_im(I,W)

% Padding image so input and output image will be same size

pad = (W - 1)/2;

img   = double(padarray(I,[pad pad],'symmetric'));
[M,N] = size(img);

variance_w = zeros(M,N);

% Compute variance within the sliding windows
for i = (pad + 1):(M - pad)
    for j = (pad + 1):(N - pad)
        
        Window = img((i-pad):(i+pad),(j-pad):(j+pad));
   
        % Extract probabilities from the histogram
        [P,C] = hist_metric(Window);
        
        G_vec = linspace(1,256,256);
        
        % Compute the mean
        mean_w = sum(G_vec.*P');
        
        % Compute the variance from by using the probabilites from the
        % historgram
        variance_w(i,j) = sum( (G_vec - mean_w).^2.*P' ) ; 
         
    end
end

Variance  = variance_w((pad + 1):(end - pad),(pad + 1):(end - pad));

end