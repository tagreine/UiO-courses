
function [scaled_image] = requant(I, Max_val)

% I       = image for requantization
% Max_val = maximum gray level value

I       = double(I);
Max_im  = max(max(I));
scale   = (Max_val-1)/Max_im;

scaled_image = uint8(I*scale);

end