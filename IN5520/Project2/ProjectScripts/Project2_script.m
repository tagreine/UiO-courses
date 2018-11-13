% Project 2 IN5520: Feature evaluation and classification 
clear;close;clc

% Load training image 
mosaic1 = load('M:\IN5520\Project2\oblig2\mosaic1_train.mat');
 
% Display training image
figure(1)
imagesc(mosaic1.mosaic1_train)
title('Training image')
colormap(gray)



% Load GLCM matrices
load('M:\IN5520\Project2\oblig2\texture1dx0dymin1.mat');
load('M:\IN5520\Project2\oblig2\texture1dx1dymin1.mat');
load('M:\IN5520\Project2\oblig2\texture1dxplus1dy0.mat');
load('M:\IN5520\Project2\oblig2\texture2dx0dymin1.mat')
load('M:\IN5520\Project2\oblig2\texture2dxmin1dymin1.mat')
load('M:\IN5520\Project2\oblig2\texture2dxplus1dy0.mat')
load('M:\IN5520\Project2\oblig2\texture2dxplus1dymin1.mat')
load('M:\IN5520\Project2\oblig2\texture3dx0dymin1.mat')
load('M:\IN5520\Project2\oblig2\texture3dxmin1dymin1.mat')
load('M:\IN5520\Project2\oblig2\texture3dxplus1dy0.mat')
load('M:\IN5520\Project2\oblig2\texture3dxplus1dymin1.mat')
load('M:\IN5520\Project2\oblig2\texture4dx0dymin1.mat')
load('M:\IN5520\Project2\oblig2\texture4dxmin1dymin1.mat')
load('M:\IN5520\Project2\oblig2\texture4dxplus1dy0.mat')
load('M:\IN5520\Project2\oblig2\texture4dxplus1dymin1.mat')

% Display GLCM matrices
figure(2)
%======================================
subplot(3,5,1)
imagesc(texture1dx0dymin1)
title('Texture 1: \Delta x = 0,\Delta y = -1')
colormap(jet)
%======================================
subplot(3,5,2)
imagesc(texture1dx1dy0)
title('Texture 1: \Delta x = 1,\Delta y = 0')
colormap(jet)
%======================================
subplot(3,5,3)
imagesc(texture1dx1dymin1)
title('Texture 1: \Delta x = 1,\Delta y = -1')
colormap(jet)
%======================================
subplot(3,5,4)
imagesc(texture2dx0dymin1)
title('Texture 2: \Delta x = 0,\Delta y = -1')
colormap(jet)
%======================================
subplot(3,5,5)
imagesc(texture2dx1dy0)
title('Texture 2: \Delta x = 1,\Delta y = 0')
colormap(jet)
%======================================
subplot(3,5,6)
imagesc(texture2dx1dymin1)
title('Texture 2: \Delta x = 1,\Delta y = -1')
colormap(jet)
%======================================
subplot(3,5,7)
imagesc(texture2dxmin1dymin1)
title('Texture 2: \Delta x = -1,\Delta y = -1')
colormap(jet)
%======================================
subplot(3,5,8)
imagesc(texture3dx0dymin1)
title('Texture 3: \Delta x = 0,\Delta y = -1')
colormap(jet)
%======================================
subplot(3,5,9)
imagesc(texture3dx1dy0)
title('Texture 3: \Delta x = 1,\Delta y = 0')
colormap(jet)
%======================================
subplot(3,5,10)
imagesc(texture3dx1dymin1)
title('Texture 3: \Delta x = 1,\Delta y = -1')
colormap(jet)
%======================================
subplot(3,5,11)
imagesc(texture3dxmin1dymin1)
title('Texture 3: \Delta x = -1,\Delta y = -1')
colormap(jet)
%======================================
subplot(3,5,12)
imagesc(texture4dx0dymin1)
title('Texture 4: \Delta x = 0,\Delta y = -1')
colormap(jet)
%======================================
subplot(3,5,13)
imagesc(texture4dx1dy0)
title('Texture 4: \Delta x = 1,\Delta y = 0')
colormap(jet)
%======================================
subplot(3,5,14)
imagesc(texture4dx1dymin1)
title('Texture 4: \Delta x = 1,\Delta y = -1')
colormap(jet)
%======================================
subplot(3,5,15)
imagesc(texture4dxmin1dymin1)
title('Texture 4: \Delta x = -1,\Delta y = -1')
colormap(jet)

% Subdivide the GLCM matrices into 4 quadrants.
% I have chosen to work with the GLCM with dx = 1 dy = -1,dx = 1 dy = 0 

%======================= dx = 1, dy = -1 ===============================================
% Texture 1
% Dividing the GLCM matrix, computed on texture 1, into 4 quadrants 
[T1Q1_1,T1Q2_1,T1Q3_1,T1Q4_1] = Quadrants(texture1dx1dymin1);
% Compute the new features  
[T1featureQ_1]                = featureQ(T1Q1_1,T1Q2_1,T1Q3_1,T1Q4_1,texture1dx1dymin1);
% Texture 2
% Dividing the GLCM matrix, computed on texture 2, into 4 quadrants 
[T2Q1_1,T2Q2_1,T2Q3_1,T2Q4_1] = Quadrants(texture2dx1dymin1);
[T2featureQ_1]                = featureQ(T2Q1_1,T2Q2_1,T2Q3_1,T2Q4_1,texture2dx1dymin1);
% Texture 3
% Dividing the GLCM matrix, computed on texture 3, into 4 quadrants 
[T3Q1_1,T3Q2_1,T3Q3_1,T3Q4_1] = Quadrants(texture3dx1dymin1);
[T3featureQ_1]                = featureQ(T3Q1_1,T3Q2_1,T3Q3_1,T3Q4_1,texture3dx1dymin1);
% Texture 4
% Dividing the GLCM matrix, computed on texture 4, into 4 quadrants 
[T4Q1_1,T4Q2_1,T4Q3_1,T4Q4_1] = Quadrants(texture4dx1dymin1);
[T4featureQ_1]                = featureQ(T4Q1_1,T4Q2_1,T4Q3_1,T4Q4_1,texture4dx1dymin1);

% Display subdivided GLCM matrices
figure(3)
%======================================
subplot(2,2,1)
imagesc(T1featureQ_1)
title('Texture 1: GLCM Quadrants')
colormap(jet)
%======================================
subplot(2,2,2)
imagesc(T2featureQ_1)
title('Texture 2: GLCM Quadrants')
colormap(jet)
%======================================
subplot(2,2,3)
imagesc(T3featureQ_1)
title('Texture 3: GLCM Quadrants')
colormap(jet)
%======================================
subplot(2,2,4)
imagesc(T4featureQ_1)
title('Texture 4: GLCM Quadrants')
colormap(jet)

%======================= dx = 1, dy = 0 ===============================================
% The same as above
% Texture 1
[T1Q1_2,T1Q2_2,T1Q3_2,T1Q4_2] = Quadrants(texture1dx1dy0);
[T1featureQ_2]                = featureQ(T1Q1_2,T1Q2_2,T1Q3_2,T1Q4_2,texture1dx1dy0);
% Texture 2
[T2Q1_2,T2Q2_2,T2Q3_2,T2Q4_2] = Quadrants(texture2dx1dy0);
[T2featureQ_2]                = featureQ(T2Q1_2,T2Q2_2,T2Q3_2,T2Q4_2,texture2dx1dy0);
% Texture 3
[T3Q1_2,T3Q2_2,T3Q3_2,T3Q4_2] = Quadrants(texture3dx1dy0);
[T3featureQ_2]                = featureQ(T3Q1_2,T3Q2_2,T3Q3_2,T3Q4_2,texture3dx1dy0);
% Texture 4
[T4Q1_2,T4Q2_2,T4Q3_2,T4Q4_2] = Quadrants(texture4dx1dy0);
[T4featureQ_2]                = featureQ(T4Q1_2,T4Q2_2,T4Q3_2,T4Q4_2,texture4dx1dy0);

% Display subdivided GLCM matrices
figure(4)
%======================================
subplot(2,2,1)
imagesc(T1featureQ_2)
title('')
colormap(jet)
%======================================
subplot(2,2,2)
imagesc(T2featureQ_2)
title('')
colormap(jet)
%======================================
subplot(2,2,3)
imagesc(T3featureQ_2)
title('')
colormap(jet)
%======================================
subplot(2,2,4)
imagesc(T4featureQ_2)
title('')
colormap(jet)

%================== histogram equalize and requantize the training image =======================
mosaic1_train_re = double(requant(mosaic1.mosaic1_train,16));
% GLCM feature computations
[featureQ1,featureQ2,featureQ3,featureQ4] = GLCM_subFeature(mosaic1_train_re,31,1,-1,16);

% Display subdivided GLCM matrices
figure(5)
%======================================
subplot(2,2,1)
imagesc(featureQ1)
title('Feature Q1')
colormap(jet)
%======================================
subplot(2,2,2)
imagesc(featureQ2)
title('Feature Q2')
colormap(jet)
%======================================
subplot(2,2,3)
imagesc(featureQ3)
title('Feature Q3')
colormap(jet)
%======================================
subplot(2,2,4)
imagesc(featureQ4)
title('Feature Q4')
colormap(jet)

%========================================Classification============================================

load('M:\IN5520\Project2\oblig2\training_mask.mat');
training_mask = double(training_mask);
% Display training mask
figure(6)
subplot(1,2,1)
imagesc(training_mask)
title('Training mask')
colormap('jet')
subplot(1,2,2)
imagesc(mosaic1_train_re.*(training_mask>0))
colormap(gca,'gray')
title('Training image: masked')

%=============================Training feature Q1 and Q2=====================================

% Two features
[featureQ1,featureQ2,featureQ3,featureQ4] = GLCM_subFeature(mosaic1_train_re,31,1,-1,16);
%features = [];features{1} = featureQ1;features{2} = featureQ2;
%features = [];features{1} = featureQ2;features{2} = featureQ3;
features = [];features{1} = featureQ3;features{2} = featureQ4; 
%features = [];features{1} = featureQ1;features{2} = featureQ4;
labels = 1:4;
[class_means,CoVar]       = MultiGaussTraining(features,training_mask,labels); 
pred_class                = MultGaussClassification(features,labels,class_means,CoVar);
mask                      = training_mask;
mask( training_mask > 0 ) = 1; 
pred_class_mask           = pred_class.*double(mask);

% Display predicted classes 
figure(7)
subplot(1,3,1)
imagesc(training_mask)
title('True Classes')
colormap('jet')
subplot(1,3,2)
imagesc(pred_class_mask)
title('Predicted classes')
colormap('jet')
caxis([0 4])

target   = training_mask(training_mask>0);
output   = pred_class_mask(pred_class_mask>0);
Conf_Mat = confusionmat(output,target);

subplot(1,3,3)
plotConfMat(Conf_Mat)
colormap('jet')


%==========================Classification on test set1 Q1 Q2================================================

load('M:\IN5520\Project2\oblig2\mosaic2_test.mat')
mosaic2_test = double(requant(mosaic2_test,16));
[featureQ1_test1,featureQ2_test1,featureQ3_test1,featureQ4_test1] = GLCM_subFeature(mosaic2_test,31,1,-1,16);
features    = [];
features{1} = featureQ1_test1;
features{2} = featureQ2_test1;
pred_class_test1      = MultGaussClassification(features,labels,class_means,CoVar);
pred_class_test1_mask = pred_class_test1.*double(mask);

% Display predicted classes 
figure(8)
subplot(1,3,1)
imagesc(training_mask)
title('True Classes')
colormap('jet')
subplot(1,3,2)
imagesc(pred_class_test1_mask)
title('Predicted classes')
colormap('jet')
caxis([0 4])

target   = training_mask(training_mask>0);
output   = pred_class_test1_mask(pred_class_test1_mask>0);
Conf_Mat = confusionmat(output,target);

subplot(1,3,3)
plotConfMat(Conf_Mat)
colormap('jet')
%==========================Classification on test set2 Q1 Q2================================================
load('M:\IN5520\Project2\oblig2\mosaic3_test.mat')
mosaic3_test = double(requant(mosaic3_test,16));
[featureQ1_test2,featureQ2_test2,featureQ3_test2,featureQ4_test2] = GLCM_subFeature(mosaic3_test,31,1,-1,16);
features    = [];
features{1} = featureQ1_test2;
features{2} = featureQ2_test2;
pred_class_test2      = MultGaussClassification(features,labels,class_means,CoVar);
pred_class_test2_mask = pred_class_test2.*double(mask);

% Display predicted classes 
figure(9)
subplot(1,3,1)
imagesc(training_mask)
title('True Classes')
colormap('jet')
subplot(1,3,2)
imagesc(pred_class_test2_mask)
title('Predicted classes')
colormap('jet')
caxis([0 4])

target   = training_mask(training_mask>0);
output   = pred_class_test2_mask(pred_class_test2_mask>0);
Conf_Mat = confusionmat(output,target);
disp(Conf_Mat)
subplot(1,3,3)
plotConfMat(Conf_Mat)
colormap('jet')
%%============================================================================================

figure(10);
subplot(3,2,1)
scatter(featureQ1(:),featureQ2(:))
xlabel('Feature 1')
ylabel('Feature 2')
grid on
subplot(3,2,2)
scatter(featureQ2(:),featureQ3(:))
xlabel('Feature 2')
ylabel('Feature 3')
grid on
subplot(3,2,3)
scatter(featureQ3(:),featureQ4(:))
xlabel('Feature 3')
ylabel('Feature 4')
grid on
subplot(3,2,4)
scatter(featureQ1(:),featureQ4(:))
xlabel('Feature 1')
ylabel('Feature 4')
grid on
subplot(3,2,5)
scatter(featureQ1(:),featureQ3(:))
xlabel('Feature 1')
ylabel('Feature 3')
grid on
subplot(3,2,6)
scatter(featureQ2(:),featureQ4(:))
xlabel('Feature 2')
ylabel('Feature 4')
grid on

%%==========================Classification on test set1 Q3 Q4================================================

features    = [];
features{1} = featureQ3_test1;
features{2} = featureQ4_test1;
pred_class_test1      = MultGaussClassification(features,labels,class_means,CoVar);
pred_class_test1_mask = pred_class_test1.*double(mask);

% Display predicted classes 
figure(8)
subplot(1,3,1)
imagesc(training_mask)
title('True Classes')
colormap('jet')
subplot(1,3,2)
imagesc(pred_class_test1_mask)
title('Predicted classes')
colormap('jet')
caxis([0 4])

target   = training_mask(training_mask>0);
output   = pred_class_test1_mask(pred_class_test1_mask>0);
Conf_Mat = confusionmat(output,target);

subplot(1,3,3)
plotConfMat(Conf_Mat)
colormap('jet')
%==========================Classification on test set2 Q3 Q4================================================

features    = [];
features{1} = featureQ3_test2;
features{2} = featureQ4_test2;
pred_class_test2      = MultGaussClassification(features,labels,class_means,CoVar);
pred_class_test2_mask = pred_class_test2.*double(mask);

% Display predicted classes 
figure(9)
subplot(1,3,1)
imagesc(training_mask)
title('True Classes')
colormap('jet')
subplot(1,3,2)
imagesc(pred_class_test2_mask)
title('Predicted classes')
colormap('jet')
caxis([0 4])

target   = training_mask(training_mask>0);
output   = pred_class_test2_mask(pred_class_test2_mask>0);
Conf_Mat = confusionmat(output,target);
disp(Conf_Mat)
subplot(1,3,3)
plotConfMat(Conf_Mat)
colormap('jet')


% Display test image
figure(1)
subplot(1,2,1)
imagesc(mosaic2_test)
title('Test image 1')
colormap(gray)
subplot(1,2,2)
figure(1)
imagesc(mosaic3_test)
title('Test image 2')
colormap(gray)







