% Project 1 IN5520
clear;close;clc

% Load images 
mosaic1 = imread('mosaic1.png');
mosaic2 = imread('mosaic2.png');

% Display images
figure(1)
subplot(1,2,1)
imagesc(mosaic1)
title('Mosaic 1')
subplot(1,2,2)
imagesc(mosaic2)
title('Mosaic 2')
colormap('gray')

% Create one subimage for each texture. Divide images into the 8 different textures
sub1 = 256; sub2=512;

text1 = mosaic1(1:sub1,1:sub1);
text2 = mosaic1(1:sub1,sub1:sub2);
text3 = mosaic1(sub1:sub2,1:sub1);
text4 = mosaic1(sub1:sub2,sub1:sub2);

text5 = mosaic2(1:sub1,1:sub1);
text6 = mosaic2(1:sub1,sub1:sub2);
text7 = mosaic2(sub1:sub2,1:sub1);
text8 = mosaic2(sub1:sub2,sub1:sub2);

% Display the images
figure(2)
% Mosaic1
subplot(2,4,1);imagesc(text1);title('Texture 1')
subplot(2,4,2);imagesc(text2);title('Texture 2')
subplot(2,4,3);imagesc(text3);title('Texture 3')
subplot(2,4,4);imagesc(text4);title('Texture 4')
% Mosaic2
subplot(2,4,5);imagesc(text5);title('Texture 5')
subplot(2,4,6);imagesc(text6);title('Texture 6')
subplot(2,4,7);imagesc(text7);title('Texture 7')
subplot(2,4,8);imagesc(text8);title('Texture 8')
colormap('gray')
%% Analyse the textures using Frequency (for frequency content, texture direction etc.), variance, homogeneity (energy) 

% Compute the frequency using 2D FFT
text1_fre = fft2(double(text1));
text2_fre = fft2(double(text2));
text3_fre = fft2(double(text3));
text4_fre = fft2(double(text4));
text5_fre = fft2(double(text5));
text6_fre = fft2(double(text6));
text7_fre = fft2(double(text7));
text8_fre = fft2(double(text8));

% Compute the energy 
text1_enrg = energy_im(text1,9);
text2_enrg = energy_im(text2,9);
text3_enrg = energy_im(text3,9);
text4_enrg = energy_im(text4,9);
text5_enrg = energy_im(text5,9);
text6_enrg = energy_im(text6,9);
text7_enrg = energy_im(text7,9);
text8_enrg = energy_im(text8,9);

% Compute the variance
text1_var = variance_im(text1,9);
text2_var = variance_im(text2,9);
text3_var = variance_im(text3,9);
text4_var = variance_im(text4,9);
text5_var = variance_im(text5,9);
text6_var = variance_im(text6,9);
text7_var = variance_im(text7,9);
text8_var = variance_im(text8,9);

figure(3)

subplot(4,8,1);imagesc(text1);title('Texture 1')
subplot(4,8,2);imagesc(text2);title('Texture 2')
subplot(4,8,3);imagesc(text3);title('Texture 3')
subplot(4,8,4);imagesc(text4);title('Texture 4')
subplot(4,8,5);imagesc(text5);title('Texture 5')
subplot(4,8,6);imagesc(text6);title('Texture 6')
subplot(4,8,7);imagesc(text7);title('Texture 7')
subplot(4,8,8);imagesc(text8);title('Texture 8')
colormap('gray')

subplot(4,8,9);imagesc(abs(fftshift(text1_fre)));colormap(gca,'jet');caxis([0 0.01*max(max(abs(text1_fre)))]);title('Frequency')
subplot(4,8,10);imagesc(abs(fftshift(text2_fre)));colormap(gca,'jet');caxis([0 0.01*max(max(abs(text2_fre)))]);title('Frequency')
subplot(4,8,11);imagesc(abs(fftshift(text3_fre)));colormap(gca,'jet');caxis([0 0.01*max(max(abs(text3_fre)))]);title('Frequency')
subplot(4,8,12);imagesc(abs(fftshift(text4_fre)));colormap(gca,'jet');caxis([0 0.01*max(max(abs(text4_fre)))]);title('Frequency')
subplot(4,8,13);imagesc(abs(fftshift(text5_fre)));colormap(gca,'jet');caxis([0 0.01*max(max(abs(text5_fre)))]);title('Frequency')
subplot(4,8,14);imagesc(abs(fftshift(text6_fre)));colormap(gca,'jet');caxis([0 0.01*max(max(abs(text6_fre)))]);title('Frequency')
subplot(4,8,15);imagesc(abs(fftshift(text7_fre)));colormap(gca,'jet');caxis([0 0.01*max(max(abs(text7_fre)))]);title('Frequency')
subplot(4,8,16);imagesc(abs(fftshift(text8_fre)));colormap(gca,'jet');caxis([0 0.01*max(max(abs(text8_fre)))]);title('Frequency')

subplot(4,8,17);imagesc(text1_enrg);colormap(gca,'jet');title('Energy')
subplot(4,8,18);imagesc(text2_enrg);colormap(gca,'jet');title('Energy')
subplot(4,8,19);imagesc(text3_enrg);colormap(gca,'jet');title('Energy')
subplot(4,8,20);imagesc(text4_enrg);colormap(gca,'jet');title('Energy')
subplot(4,8,21);imagesc(text5_enrg);colormap(gca,'jet');title('Energy')
subplot(4,8,22);imagesc(text6_enrg);colormap(gca,'jet');title('Energy')
subplot(4,8,23);imagesc(text7_enrg);colormap(gca,'jet');title('Energy')
subplot(4,8,24);imagesc(text8_enrg);colormap(gca,'jet');title('Energy')

subplot(4,8,25);imagesc(text1_var);colormap(gca,'jet');title('Variance')
subplot(4,8,26);imagesc(text2_var);colormap(gca,'jet');title('Variance')
subplot(4,8,27);imagesc(text3_var);colormap(gca,'jet');title('Variance')
subplot(4,8,28);imagesc(text4_var);colormap(gca,'jet');title('Variance')
subplot(4,8,29);imagesc(text5_var);colormap(gca,'jet');title('Variance')
subplot(4,8,30);imagesc(text6_var);colormap(gca,'jet');title('Variance')
subplot(4,8,31);imagesc(text7_var);colormap(gca,'jet');title('Variance')
subplot(4,8,32);imagesc(text8_var);colormap(gca,'jet');title('Variance')

%% Preprocessing for GLCM, using histogram equalization and requantization
  
% Histogram equalization. Do we need to uniformly distribute the pixels, or
% are the uniform enough? Lets check the histograms

[text1_hist,text1_cumhist] = hist_metric(text1);
[text2_hist,text2_cumhist] = hist_metric(text2);
[text3_hist,text3_cumhist] = hist_metric(text3);
[text4_hist,text4_cumhist] = hist_metric(text4);
[text5_hist,text5_cumhist] = hist_metric(text5);
[text6_hist,text6_cumhist] = hist_metric(text6);
[text7_hist,text7_cumhist] = hist_metric(text7);
[text8_hist,text8_cumhist] = hist_metric(text8);

% Display the images with corresponding histograms
figure(4)

subplot(3,8,1);imagesc(text1);title('Texture 1')
subplot(3,8,2);imagesc(text2);title('Texture 2')
subplot(3,8,3);imagesc(text3);title('Texture 3')
subplot(3,8,4);imagesc(text4);title('Texture 4')
subplot(3,8,5);imagesc(text5);title('Texture 5')
subplot(3,8,6);imagesc(text6);title('Texture 6')
subplot(3,8,7);imagesc(text7);title('Texture 7')
subplot(3,8,8);imagesc(text8);title('Texture 8')
colormap('gray')

subplot(3,8,9);plot(text1_hist);title('Histogram')
subplot(3,8,10);plot(text2_hist);title('Histogram')
subplot(3,8,11);plot(text3_hist);title('Histogram')
subplot(3,8,12);plot(text4_hist);title('Histogram')
subplot(3,8,13);plot(text5_hist);title('Histogram')
subplot(3,8,14);plot(text6_hist);title('Histogram')
subplot(3,8,15);plot(text7_hist);title('Histogram')
subplot(3,8,16);plot(text8_hist);title('Histogram')

subplot(3,8,17);plot(text1_cumhist);title('Cumulative')
subplot(3,8,18);plot(text2_cumhist);title('Cumulative')
subplot(3,8,19);plot(text3_cumhist);title('Cumulative')
subplot(3,8,20);plot(text4_cumhist);title('Cumulative')
subplot(3,8,21);plot(text5_cumhist);title('Cumulative')
subplot(3,8,22);plot(text6_cumhist);title('Cumulative')
subplot(3,8,23);plot(text7_cumhist);title('Cumulative')
subplot(3,8,24);plot(text8_cumhist);title('Cumulative')

% From the histograms, I conclude that textures should have
% histogram equalization applied.
Max1     = double(max(max(text1)));
Max2     = double(max(max(text2)));
Max3     = double(max(max(text3)));
Max4     = double(max(max(text4)));
Max5     = double(max(max(text5)));
Max6     = double(max(max(text6)));
Max7     = double(max(max(text7)));
Max8     = double(max(max(text8)));

% Compute the histogram equalization using maximum number of gray levels in
% the images 
text1_eq  = histeq(text1,Max1);
text2_eq  = histeq(text2,Max2);
text3_eq  = histeq(text3,Max3);
text4_eq  = histeq(text4,Max4);
text5_eq  = histeq(text5,Max5);
text6_eq  = histeq(text6,Max6);
text7_eq  = histeq(text7,Max7);
text8_eq  = histeq(text8,Max8);

[text1_hist,text1_cumhist] = hist_metric(text1_eq);
[text2_hist,text2_cumhist] = hist_metric(text2_eq);
[text3_hist,text3_cumhist] = hist_metric(text3_eq);
[text4_hist,text4_cumhist] = hist_metric(text4_eq);
[text5_hist,text5_cumhist] = hist_metric(text5_eq);
[text6_hist,text6_cumhist] = hist_metric(text6_eq);
[text7_hist,text7_cumhist] = hist_metric(text7_eq);
[text8_hist,text8_cumhist] = hist_metric(text8_eq);

% Display the images with corresponding histograms and equalized histograms
figure(5)

subplot(3,8,1);imagesc(text1_eq);title('Texture 1')
subplot(3,8,2);imagesc(text2_eq);title('Texture 2')
subplot(3,8,3);imagesc(text3_eq);title('Texture 3')
subplot(3,8,4);imagesc(text4_eq);title('Texture 4')
subplot(3,8,5);imagesc(text5_eq);title('Texture 5')
subplot(3,8,6);imagesc(text6_eq);title('Texture 6')
subplot(3,8,7);imagesc(text7_eq);title('Texture 7')
subplot(3,8,8);imagesc(text8_eq);title('Texture 8')
colormap('gray')

subplot(3,8,9);plot(text1_hist);title('Histogram')
subplot(3,8,10);plot(text2_hist);title('Histogram')
subplot(3,8,11);plot(text3_hist);title('Histogram')
subplot(3,8,12);plot(text4_hist);title('Histogram')
subplot(3,8,13);plot(text5_hist);title('Histogram')
subplot(3,8,14);plot(text6_hist);title('Histogram')
subplot(3,8,15);plot(text7_hist);title('Histogram')
subplot(3,8,16);plot(text8_hist);title('Histogram')

subplot(3,8,17);plot(text1_cumhist);title('Cumulative')
subplot(3,8,18);plot(text2_cumhist);title('Cumulative')
subplot(3,8,19);plot(text3_cumhist);title('Cumulative')
subplot(3,8,20);plot(text4_cumhist);title('Cumulative')
subplot(3,8,21);plot(text5_cumhist);title('Cumulative')
subplot(3,8,22);plot(text6_cumhist);title('Cumulative')
subplot(3,8,23);plot(text7_cumhist);title('Cumulative')
subplot(3,8,24);plot(text8_cumhist);title('Cumulative')

% Requantize the images to reduce gray levels

text1_re  = requant(text1_eq,16);
text2_re  = requant(text2_eq,16);
text3_re  = requant(text3_eq,16);
text4_re  = requant(text4_eq,16);
text5_re  = requant(text5_eq,16);
text6_re  = requant(text6_eq,16);
text7_re  = requant(text7_eq,16);
text8_re  = requant(text8_eq,16);

% Display the images after requantization
figure(5)
% Mosaic1
subplot(2,4,1);imagesc(text1_re)
subplot(2,4,2);imagesc(text2_re)
subplot(2,4,3);imagesc(text3_re)
subplot(2,4,4);imagesc(text4_re)
% Mosaic2
subplot(2,4,5);imagesc(text5_re)
subplot(2,4,6);imagesc(text6_re)
subplot(2,4,7);imagesc(text7_re)
subplot(2,4,8);imagesc(text8_re)
colormap('gray')

%% GLCM matrix computations 

% Offset for GLCM computation
dx = 1;
dy = 1;

[text1_GLCM_H,text1_GLCM_V,text1_GLCM_EE,text1_GLCM_VV] = GLCM_gen(text1_re,dx,dy);
[text2_GLCM_H,text2_GLCM_V,text2_GLCM_EE,text2_GLCM_VV] = GLCM_gen(text2_re,dx,dy);
[text3_GLCM_H,text3_GLCM_V,text3_GLCM_EE,text3_GLCM_VV] = GLCM_gen(text3_re,dx,dy);
[text4_GLCM_H,text4_GLCM_V,text4_GLCM_EE,text4_GLCM_VV] = GLCM_gen(text4_re,dx,dy);
[text5_GLCM_H,text5_GLCM_V,text5_GLCM_EE,text5_GLCM_VV] = GLCM_gen(text5_re,dx,dy);
[text6_GLCM_H,text6_GLCM_V,text6_GLCM_EE,text6_GLCM_VV] = GLCM_gen(text6_re,dx,dy);
[text7_GLCM_H,text7_GLCM_V,text7_GLCM_EE,text7_GLCM_VV] = GLCM_gen(text7_re,dx,dy);
[text8_GLCM_H,text8_GLCM_V,text8_GLCM_EE,text8_GLCM_VV] = GLCM_gen(text8_re,dx,dy);

% Display the GLCM images. If the texture has a clear orientation select ? according to this.
figure(6)

subplot(2,8,1);imagesc(text1_re)
subplot(2,8,2);imagesc(text2_re)
subplot(2,8,3);imagesc(text3_re)
subplot(2,8,4);imagesc(text4_re)
subplot(2,8,5);imagesc(text5_re)
subplot(2,8,6);imagesc(text6_re)
subplot(2,8,7);imagesc(text7_re)
subplot(2,8,8);imagesc(text8_re)
colormap('gray')
% texture 1: no dominant direction, we use isotropic GLCM
subplot(2,8,9);imagesc( ((text1_GLCM_H+text1_GLCM_V+text1_GLCM_EE+text1_GLCM_VV)/4) );colormap(gca,'jet');colorbar('SouthOutside');title('GLCM')
% texture 2: no dominant direction, we use isotropic GLCM
subplot(2,8,10);imagesc( ((text2_GLCM_H+text2_GLCM_V+text2_GLCM_EE+text2_GLCM_VV)/4) );colormap(gca,'jet');colorbar('SouthOutside');title('GLCM')
% texture 3: sub-vertical dominant direction, we use GLCM in vertical and
% in 45/315deg direction
subplot(2,8,11);imagesc( ((text3_GLCM_H+text3_GLCM_EE)/2) );colormap(gca,'jet');colorbar('SouthOutside');title('GLCM')
% texture 4: no dominant direction, we use isotropic GLCM
subplot(2,8,12);imagesc( ((text4_GLCM_H+text4_GLCM_V+text4_GLCM_EE+text4_GLCM_VV)/4) );colormap(gca,'jet');colorbar('SouthOutside');title('GLCM')
% texture 5: Vertical and horizontal dominant direction, we use GLCM
% vertical and horizontal
subplot(2,8,13);imagesc( ((text5_GLCM_H+text5_GLCM_V)/2) );colormap(gca,'jet');colorbar('SouthOutside');title('GLCM')
% texture 6: Vertical and horizontal dominant direction, we use GLCM
% vertical and horizontal
subplot(2,8,14);imagesc( ((text6_GLCM_H+text6_GLCM_V)/2) );colormap(gca,'jet');colorbar('SouthOutside');title('GLCM')
% texture 7: Vertical dominant direction, we use GLCM vertical
subplot(2,8,15);imagesc( (text7_GLCM_V) );colormap(gca,'jet');colorbar('SouthOutside');title('GLCM')
% texture 8: no dominant direction, we use isotropic GLCM
subplot(2,8,16);imagesc( ((text8_GLCM_H+text8_GLCM_V+text8_GLCM_EE+text8_GLCM_VV)/4) );colormap(gca,'jet'),colorbar('SouthOutside');title('GLCM')


%% GLCM feature computation testing 

% Now we try the GLCM features on the data

% Offset for GLCM computation
dx = 1;
dy = 1;
% Chose directions and merges for improving GLCM directional dependence: 
% a = isotropic; b = North-South; c = East-West; d = diagonal-right; 
% e = diagonal-left f = b+c; g = b+d; h = b+e; i = c+d; j = c+e; 
% k = d+e;

% Window size
W = 31;

% texture 1: no dominant direction, we use isotropic GLCM --> 'a'
[text1_features] = GLCM_features_slideW(text1_re,W,dx,dy,'a');
% texture 2: no dominant direction, we use isotropic GLCM --> 'a'
[text2_features] = GLCM_features_slideW(text2_re,W,dx,dy,'a');
% texture 3: sub-vertical towards the right dominant direction, we use GLCM --> 'g' = b + d 
[text3_features] = GLCM_features_slideW(text3_re,W,dx,dy,'g');
% texture 4: no dominant direction, we use isotropic GLCM --> 'a'
[text4_features] = GLCM_features_slideW(text4_re,W,dx,dy,'a');
% texture 5: Vertical and horizontal dominant direction, we use GLCM --> 'f' = b + c 
[text5_features] = GLCM_features_slideW(text5_re,W,dx,dy,'f');
% texture 6: Vertical and horizontal dominant direction, we use GLCM --> 'f' = b + c 
[text6_features] = GLCM_features_slideW(text6_re,W,dx,dy,'f');
% texture 7: Vertical dominant direction, we use GLCM --> 'b'
[text7_features] = GLCM_features_slideW(text7_re,W,dx,dy,'b');
% texture 8: no dominant direction, we use isotropic GLCM --> 'a'
[text8_features] = GLCM_features_slideW(text8_re,W,dx,dy,'a');


figure(7)

subplot(4,8,1);imagesc(text1_re);title('Texture 1')
subplot(4,8,2);imagesc(text2_re);title('Texture 2')
subplot(4,8,3);imagesc(text3_re);title('Texture 3')
subplot(4,8,4);imagesc(text4_re);title('Texture 4')
subplot(4,8,5);imagesc(text5_re);title('Texture 5')
subplot(4,8,6);imagesc(text6_re);title('Texture 6')
subplot(4,8,7);imagesc(text7_re);title('Texture 7')
subplot(4,8,8);imagesc(text8_re);title('Texture 8')
colormap('gray')

subplot(4,8,9);imagesc( (text1_features.Con) );colormap(gca,'jet');title('Contrast');colorbar('SouthOutside')
subplot(4,8,10);imagesc( (text2_features.Con) );colormap(gca,'jet');title('Contrast');colorbar('SouthOutside')
subplot(4,8,11);imagesc( (text3_features.Con) );colormap(gca,'jet');title('Contrast');colorbar('SouthOutside')
subplot(4,8,12);imagesc( (text4_features.Con) );colormap(gca,'jet');title('Contrast');colorbar('SouthOutside')
subplot(4,8,13);imagesc( (text5_features.Con) );colormap(gca,'jet');title('Contrast');colorbar('SouthOutside')
subplot(4,8,14);imagesc( (text6_features.Con) );colormap(gca,'jet');title('Contrast');colorbar('SouthOutside')
subplot(4,8,15);imagesc( (text7_features.Con) );colormap(gca,'jet');title('Contrast');colorbar('SouthOutside')
subplot(4,8,16);imagesc( (text8_features.Con) );colormap(gca,'jet');title('Contrast');colorbar('SouthOutside')

subplot(4,8,17);imagesc( (text1_features.IDM) );colormap(gca,'jet');title('IDM');colorbar('SouthOutside')
subplot(4,8,18);imagesc( (text2_features.IDM) );colormap(gca,'jet');title('IDM');colorbar('SouthOutside')
subplot(4,8,19);imagesc( (text3_features.IDM) );colormap(gca,'jet');title('IDM');colorbar('SouthOutside')
subplot(4,8,20);imagesc( (text4_features.IDM) );colormap(gca,'jet');title('IDM');colorbar('SouthOutside')
subplot(4,8,21);imagesc( (text5_features.IDM) );colormap(gca,'jet');title('IDM');colorbar('SouthOutside')
subplot(4,8,22);imagesc( (text6_features.IDM) );colormap(gca,'jet');title('IDM');colorbar('SouthOutside')
subplot(4,8,23);imagesc( (text7_features.IDM) );colormap(gca,'jet');title('IDM');colorbar('SouthOutside')
subplot(4,8,24);imagesc( (text8_features.IDM) );colormap(gca,'jet');title('IDM');colorbar('SouthOutside')

subplot(4,8,25);imagesc( (text1_features.CSH) );colormap(gca,'jet');title('Cluster shade');colorbar('SouthOutside')
subplot(4,8,26);imagesc( (text2_features.CSH) );colormap(gca,'jet');title('Cluster shade');colorbar('SouthOutside')
subplot(4,8,27);imagesc( (text3_features.CSH) );colormap(gca,'jet');title('Cluster shade');colorbar('SouthOutside')
subplot(4,8,28);imagesc( (text4_features.CSH) );colormap(gca,'jet');title('Cluster shade');colorbar('SouthOutside')
subplot(4,8,29);imagesc( (text5_features.CSH) );colormap(gca,'jet');title('Cluster shade');colorbar('SouthOutside')
subplot(4,8,30);imagesc( (text6_features.CSH) );colormap(gca,'jet');title('Cluster shade');colorbar('SouthOutside')
subplot(4,8,31);imagesc( (text7_features.CSH) );colormap(gca,'jet');title('Cluster shade');colorbar('SouthOutside')
subplot(4,8,32);imagesc( (text8_features.CSH) );colormap(gca,'jet');title('Cluster shade');colorbar('SouthOutside')



%% GLCM feature computation and segmentation

% Display images
figure(8)

subplot(1,2,1)
imagesc(mosaic1)
subplot(1,2,2)
imagesc(mosaic2)
colormap('gray')

% Now we shal try to segment out the textures using different GLCM features.
% First we preprocess our data with:

% Histogram equalization
MaxM1     = double(max(max(text2)));
MaxM2     = double(max(max(text3)));

mosaic1_eq = histeq(mosaic1,MaxM1);
mosaic2_eq = histeq(mosaic2,MaxM2);

% Image requantization
mosaic1_re  = requant(mosaic1_eq,16);
mosaic2_re  = requant(mosaic2_eq,16);

% Offset for GLCM computation
dx = 1;
dy = 1;

% The GLCM_features_slideW creates 5 features within a structure: variance,
% contrast, IDM, entropy and cluster shade.
% Chose directions and merges for improving GLCM directional dependence: 
% a = isotropic; b = North-South; c = East-West; d = diagonal-right; 
% e = diagonal-left f = b+c; g = b+d; h = b+e; i = c+d; j = c+e; 
% k = d+e;

% In this case I used isotropic GLCM
[mosaic1_features] = GLCM_features_slideW(mosaic1_re,W,dx,dy,'a');
[mosaic2_features] = GLCM_features_slideW(mosaic2_re,W,dx,dy,'a');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Global thresholding of Mosaic 1 image features

% Seperation of texture 2 using contrast
threshold1 = zeros(size(mosaic1_features.Con));
threshold1(mosaic1_features.Con < 7.5) = 1;
figure;imagesc(threshold1)

% Seperation of texture 3 using cluster shade
threshold2 = zeros(size(mosaic1_features.CSH));
threshold2(mosaic1_features.CSH < -100) = 1;
figure;imagesc(threshold2)

% Seperation of texture 4 inverse difference matrix
threshold3 = zeros(size(mosaic1_features.IDM));
threshold3(mosaic1_features.IDM < 0.335) = 1;
figure;imagesc(threshold3)

% 
threshold_test = zeros(size(mosaic1_features.Con));
threshold_test(mosaic1_features.Con < 14) = 1;
figure;imagesc(threshold_test)

% 
threshold_test2 = zeros(size(mosaic1_features.IDM));
threshold_test2(mosaic1_features.IDM < 0.39) = 1;
figure;imagesc(threshold_test2)

figure;imagesc(threshold_test.*threshold_test2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Global thresholding of Mosaic 2 image features

% Seperation of texture 6 using contrast
threshold4 = zeros(size(mosaic2_features.Con));
threshold4(mosaic2_features.Con < 11.5) = 1;
figure;imagesc(threshold4)

% Seperation of texture 8
threshold5 = zeros(size(mosaic2_features.CSH));
threshold5(mosaic2_features.CSH > 250) = 1;
figure;imagesc(threshold5)

% Seperation of texture 6
threshold6 = zeros(size(mosaic2_features.IDM));
threshold6(mosaic2_features.IDM > 0.40) = 1;
figure;imagesc(threshold6)

% 
threshold_test3 = zeros(size(mosaic2_features.Con));
threshold_test3(mosaic2_features.Con > 11.5) = 1;
figure;imagesc(threshold_test3)

threshold_test4 = zeros(size(mosaic2_features.CSH));
threshold_test4(mosaic2_features.CSH < 120) = 1;
figure;imagesc(threshold_test4)

figure;imagesc(threshold_test3.*threshold_test4)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(9)

subplot(2,3,1)
imagesc(threshold1);title('A: GLCM Contrast')
%colormap('gray')

subplot(2,3,2)
imagesc(threshold2);title('B: GLCM Cluster Shade')
%colormap('gray')

subplot(2,3,3)
imagesc(threshold3);title('C: GLCM IDM')
%colormap('gray')

subplot(2,3,4)
imagesc(threshold4);title('D: GLCM Inertia')
%colormap('gray')

subplot(2,3,5)
imagesc(threshold5);title('E: GLCM Cluster Shade')
%colormap('gray')

subplot(2,3,6)
imagesc(threshold6);title('F: GLCM IDM')
%colormap('gray')

% Segmentation of textures using isotropic GLCM 

figure(10)

subplot(2,3,1)
imagesc(mosaic1.*uint8(threshold1));title('A: Texture 2')
colormap('gray')

subplot(2,3,2)
imagesc(mosaic1.*uint8(threshold2));title('B: Texture 3')
colormap('gray')

subplot(2,3,3)
imagesc(mosaic1.*uint8(threshold3));title('C: Texture 4')
colormap('gray')

subplot(2,3,4)
imagesc(mosaic2.*uint8(threshold4));title('D: Texture 6')
colormap('gray')

subplot(2,3,5)
imagesc(mosaic2.*uint8(threshold5));title('E: Texture 8')
colormap('gray')

subplot(2,3,6)
imagesc(mosaic2.*uint8(threshold6));title('F: Texture 6')
colormap('gray')

figure(11)

subplot(1,2,1)
imagesc(mosaic1.*uint8(threshold_test.*threshold_test2));title('A: Texture 1')
colormap('gray')

subplot(1,2,2)
imagesc(mosaic2.*uint8(threshold_test3.*threshold_test4));title('B: Texture 7')
colormap('gray')
