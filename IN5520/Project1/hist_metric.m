

function [hist_p,cumhist] = hist_metric(img)
    
    % Computes the normalized histogram and the cumulative histogram
    
    % The historgram
    GrayL = 256;
    hist = zeros(GrayL,1);
    
    for i=1:GrayL
        
        hist(i) = sum(sum(img == i-1));
    
    end
    
    hist_p = hist./numel(img);
    
    % The cumulative histogram
    cumhist = zeros(1,GrayL);
    cumhist(1) = hist_p(1);
    
    for i=2:GrayL
        
        cumhist(i) = cumhist(i-1) + hist_p(i);
        
    end  
end