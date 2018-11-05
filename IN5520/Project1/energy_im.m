
function [Energy] = energy_im(I,W)

    % Padding image so input and output image will be same size

    pad = (W - 1)/2;

    img   = double(padarray(I,[pad pad],'symmetric'));
    [M,N] = size(img);
    
    % Define the energy
    Energy_w = zeros(M,N);

    % Compute energy within the sliding windows 
    for i = (pad + 1):(M - pad)
        for j = (pad + 1):(N - pad)
        
            Window = img((i-pad):(i+pad),(j-pad):(j+pad));
            
            % Extract probabilities from the histogram
            [P,C] = hist_metric(Window);
            
            % Compute the energy from by using the probabilites from the
            % historgram
            Energy_w(i,j) = sum(P.^2);               
         
        end
    end

    Energy  = Energy_w((pad + 1):(end - pad),(pad + 1):(end - pad));

end