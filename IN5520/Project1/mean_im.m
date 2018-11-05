
function [Mean] = mean_im(I,W)

% Padding image so input and output image will be same size
pad = (W - 1)/2;
img   = double(padarray(I,[pad pad],'symmetric'));

% Size of image
[M,N] = size(img);

% Define the mean 
Mean_w = zeros(M,N);

% Computing the mean in a sliding window
for i = (pad + 1):(M - pad)
    for j = (pad + 1):(N - pad)
        
        Window = img((i-pad):(i+pad),(j-pad):(j+pad));
        
        % Compute the histogram intesity levels
        [P,C] = hist_metric(Window);
        
        G_vec = linspace(1,256,256);
        
        % Compute the mean
        Mean_w(i,j) = sum(G_vec.*P');
         
    end
end

Mean  = Mean_w((pad + 1):(end - pad),(pad + 1):(end - pad));

end