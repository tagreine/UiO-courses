function [Gauss] = MultiGaussSVD(pixel_val,sigma,mu,num_labels,num_features)

    % Compute the discrimination function
    [V,D,UT] = svd(sigma);
    D_      = D;
    [m,n] = size(D);
    for i = 1:m
        for j= 1:n
            if i == j && abs(D(i,j))>0 
            
                D_(i,j) = 1/D(i,j);
                
            end
        end
    end
    
    E = UT*D_*V;
    Gauss = -(1/2)*((pixel_val - mu)')*E*(pixel_val - mu) - (num_features/2)*log(2*pi) - (1/2)*log(det(sigma)) + log(1/num_labels);
    
end