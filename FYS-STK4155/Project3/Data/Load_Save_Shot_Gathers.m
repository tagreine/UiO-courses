%% ============================ Load synthetic seismic =================================
clear;close;clc
addpath("M:\FYS-STK4155\Project3\segymat-master\segymat-master")
addpath("M:\FYS-STK4155\Project3\1997_2.5D_shots.segy")

[Shots,SegyTraceHeaders,SegyHeader]=ReadSegy('1997_2.5D_shots.segy');

[A,B] = size(Shots);

start = 10;
stopp = 30;

imagesc(Shots(1:250,256*(start-1):256*stopp))
colormap(gray)
caxis([-0.1 0.1])

%% ============================= Make training data set ==========================
% Random draw individual shots from the data

nr_shots = stopp - start;

nr = randi([1 B/256],1,nr_shots);


Shots_training = zeros(nr_shots,250,256);

for j = 1:nr_shots

    Shots_training(j,:,:) = Shots(1:250,( 256*(nr(j) - 1) + 1 ) : 256*nr(j) );
    
end


% Apply gain

[K,M,N] = size(Shots_training);

t2 = ([0:(M-1)]*0.004).^1.1;
%figure;plot(t2)
Shots_training_g = zeros(K,M,N);

for k = 1:K
    for i = 1:M

        Shots_training_g(k,i,:) = Shots_training(k,i,:)*t2(i);
        
    end
end


% Normalize the data

Shots_training_gn = zeros(K,M,200);

for k = 1:K
    Shots_training_gn(k,:,:) = Shots_training_g(k,:,1:200)/max(max(abs(Shots_training_g(k,:,1:200))));
end

figure(2);

for i = 1:20
    subplot(2,10,i)
    sh     = squeeze(Shots_training_gn(i,:,:));
    imagesc(sh)
    colormap(gray)
    caxis([-0.1 0.1])
end


%save('Training_data.mat','Shots_training_gn');



