clc;
clear all;
close all;
disp('This code may take up to 7 minutes to run, thanks for your patience.');
disp('___________________________________________________________________________________________________________________________________________');
%% Loading and processing the initial images to learn, validate, and test the model
location = input('Please provide the directory containing the input images used by coders to learn the model: \n', 's'); % ask the user to enter the address of the directory that includes 2 sub-directories (0 and 1), these images are the ones that we got after visually validating the initial images that we were given.
t_start1 = tic; % to get the execution time associated with loading and processing the images
Images_0 = dir(strcat(location,'\0\*.png'));
Images_1 = dir(strcat(location,'\1\*.png'));
m_0 = numel(Images_0); % number of images in directory 0 
m_1 = numel(Images_1); % number of images in directory 1
% X_0 = zeros(101,101,1,m_0);
% X_1 = zeros(101,101,1,m_1);
for i=1:m_0 % reading images in sub-directory 0
     fullFileName_0 = strcat(Images_0(i).folder,'\',Images_0(i).name);
     I = imread(fullFileName_0);
     I = im2double(I); % converting images to double precision
     X_0(:,:,:,i) = I; % storing images into a 4-D matrix 
end
for j=1:m_1 % reading images in sub-directory 1
     fullFileName_1 = strcat(Images_1(j).folder,'\',Images_1(j).name);
     J = imread(fullFileName_1);
     J = im2double(J); % converting images to double precision
     X_1(:,:,:,j) = J; % storing images into a 4-D matrix
end
%% Labeling the images; Target Matrices (1 is for class 1 (worm) and 0 is for class 2 (no worm))
% for p1 = 1:m_0
%     T_0(p1,1) = 0;
%     T_0(p1,2) = 1;
% end
% for p2 = 1:m_1
%     T_1(p2,1) = 1;
%     T_1(p2,2) = 0;
% end
T_0 = zeros(m_0,1);
T_1 = ones(m_1,1);
t_end1 = toc(t_start1); % end time associated with loading and processing the images
%% Training set
X_train = cat(4,X_0(:,:,:,(1:(0.6*size(X_0,4)))),X_1(:,:,:,(1:(0.6*size(X_1,4)))));
T_train = [T_0((1:(0.6*size(T_0,1))),:);T_1((1:(0.6*size(T_1,1))),:)];
%% Validation set
X_valid = cat(4,X_0(:,:,:,((0.6*size(X_0,4)+1):(0.8*size(X_0,4)))),X_1(:,:,:,((0.6*size(X_1,4)+1):(0.8*size(X_1,4)))));
T_valid = categorical([T_0(((0.6*size(T_0,1)+1):(0.8*size(T_0,1))),:);T_1(((0.6*size(T_1,1)+1):(0.8*size(T_1,1))),:)]);
%% Test set
X_test = cat(4,X_0(:,:,:,((0.8*size(X_0,4)+1):size(X_0,4))),X_1(:,:,:,((0.8*size(X_1,4)+1):size(X_1,4))));
T_test = [T_0(((0.8*size(T_0,1)+1):size(T_0,1)),:);T_1(((0.8*size(T_1,1)+1):size(T_1,1)),:)];
%% Input layer
input_layer = imageInputLayer([101 101 1],'DataAugmentation','none',...
    'Normalization','zscore','Name','Input_Layer');
%% First convolution and pooling layers
m1 = 5;
n1 = 16;
convolution_layer1 = convolution2dLayer(m1,n1,'Stride',2,'Padding',0,'BiasLearnRateFactor',1,...
    'NumChannels',1,'WeightLearnRateFactor',1,'WeightL2Factor',1,...
    'BiasL2Factor',1,'BiasInitializer','ones','Name','Convolution_Layer1');
convolution_layer1.Weights = rand([m1 m1 1 n1]);
convolution_layer1.Bias = rand([1 1 n1]);
batch_norm_layer1 = batchNormalizationLayer('Name','Batch_Norm_Conv_Layer1'); % Batch normalization actually has 2 learnable parameters just like weights to adjust the mean and the variance it uses to normalize the data so that it can normalize the data for better results
relu_convolution_layer1 = reluLayer('Name','Relu_Conv_Layer1'); % Passing it through a Relu layer (for non-linearity)
% localnormlayer1 = crossChannelNormalizationLayer(3,'Name',...localnorm1','Alpha',0.0001,'Beta',0.75,'K',2); --> Another option that could have been used instead of batchnormalization (details unknown)
pooling_layer1 = maxPooling2dLayer(4,'Stride',3,'Name','Pooling_Layer1','Padding',0);
%% Second convolution and pooling layers
m2 = 4;
n2 = 32;
convolution_layer2 = convolution2dLayer(m2,n2,'Stride',1,'Padding',0,'BiasLearnRateFactor',1,...
    'NumChannels',n1,'WeightLearnRateFactor',1,'WeightL2Factor',1,...
    'BiasL2Factor',1,'BiasInitializer','ones','Name','Convolution_Layer2');
convolution_layer2.Weights = rand([m2 m2 n1 n2]);
convolution_layer2.Bias = rand([1 1 n2]);
batch_norm_layer2 = batchNormalizationLayer('Name','Batch_Norm_Conv_Layer2');
relu_convolution_layer2 = reluLayer('Name','Relu_Conv_Layer2');
pooling_layer2 = maxPooling2dLayer(3,'Stride',2,'Name','Pooling_Layer2','Padding',0);
%% Hidden layer
% flat_layer = flattenLayer('Name','flat');
n3 = 5000;
hidden_layer = fullyConnectedLayer(n3,'WeightLearnRateFactor',1,'BiasLearnRateFactor',1,...
    'BiasInitializer','ones','WeightL2Factor',1,'BiasL2Factor',1,'Name','Hidden_Layer');
hidden_layer.Weights = rand([n3 1152]); % 5000 is the output from fullyconnectedlayer1 layer to the next fullyconnectedlayer2
hidden_layer.Bias = rand([n3 1]);
batch_norm_layer3 = batchNormalizationLayer('Name','Batch_Norm_Hidden_Layer');
relu_hidden_layer = reluLayer('Name','Relu_Hidden_Layer');
%% Output layer
output_layer = fullyConnectedLayer(2,'WeightLearnRateFactor',1,...
    'BiasLearnRateFactor',1,'BiasInitializer','ones','WeightL2Factor',1,'BiasL2Factor',1,...
    'Name','Output_Layer');
output_layer.Weights = rand([2 n3]); % 5000 inputs and 2 outputs for softmax layer due to 2 classes (worm + noworm)
output_layer.Bias = rand([2 1]);
batch_norm_layer4 = batchNormalizationLayer('Name','Batch_Norm_Output_Layer');
softmax_output_layer = softmaxLayer('Name','Softmax_Output_Layer');
result_output_layer = classificationLayer('Name','Results_Output_Layer');
%%
options = trainingOptions('rmsprop','LearnRateSchedule','none','LearnRateDropFactor',0.99,...
    'LearnRateDropPeriod',4,'SquaredGradientDecayFactor',0.99,'ValidationFrequency',60,...
    'ValidationPatience',100,'ValidationData',{X_valid,T_valid},...
    'L2Regularization',0,'MaxEpochs',50,'Shuffle','once','Plots','training-progress',...
    'ExecutionEnvironment','multi-gpu','MiniBatchSize',30,'Verbose',1,'CheckpointPath',...
    'C:\Users\ykocoglu\Desktop\ff\checkpoints','InitialLearnRate',0.01); % optimizer = 'sgdm',InitialLearnRate = 0.5, 'Momentum' = ,0.99

%% Order of layers; CNN structure
layers =[input_layer,convolution_layer1,batch_norm_layer1,relu_convolution_layer1,...
    pooling_layer1,convolution_layer2,batch_norm_layer2,relu_convolution_layer2,pooling_layer2,...
    hidden_layer,batch_norm_layer3,relu_hidden_layer,output_layer,batch_norm_layer4,softmax_output_layer,result_output_layer];


%% Train the network
train_images = X_train;
train_labels = categorical(T_train);
net.trainParam.goal = 1e-3; 
model = trainNetwork(train_images,train_labels,layers,options);
% tic-toc to record testing time for all 3000 images
tic;
test_images = X_test;
[Y,scores] = classify(model,test_images);
score = sum((Y == categorical(T_test)))/size(T_test,1);
% store_results(i,1) = score;
toc;
% Display all the test accuracy obtained (20 of them)
score

