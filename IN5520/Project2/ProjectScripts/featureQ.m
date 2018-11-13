
function [FeatureQ] = featureQ(Q1,Q2,Q3,Q4,I)

Isum = sum(sum(I));
Q1sum = sum(sum(Q1));
Q2sum = sum(sum(Q2));
Q3sum = sum(sum(Q3));
Q4sum = sum(sum(Q4));

FeatureQ = [Q1sum/Isum,Q2sum/Isum;Q3sum/Isum,Q4sum/Isum];

end