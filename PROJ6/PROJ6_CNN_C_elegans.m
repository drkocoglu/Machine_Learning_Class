% PROJECT#6 - C.elegans image recognition

%% Sam's comments on the C. elegans data:

% Training set has a total of 11000 images (5500 per class)
% Test set has a total of 3000 images (1500 per class)
% Focused on adding examples of dead worms
% Presence of some image augmentation
% Prepared this with the mind of getting a 1 if most of the worm is in the picture and 0 if a negligible amount of the worm is in the picture.
% Please aim to not alter the number of the images if you plan to modify the set in any way

%% Notes from Andrew Ng's Deeplearning course and more resources for understanding the CNN application

% Steps to follow to build a CNN algorithm:
% 1) Choose filter size and number of filters (might or might not be of same size for each filter) --> number of filters will change the 3rd dimension of outputlayer from *1 to number of filters
% 2) Apply convolution with/without padding (adding 0s to the edges of the images) and/or striding (adding a specified number of jumps in convolution calculation). If Max pooling is applied usually padding is not used (?).
% 3) Choose to "apply/do not apply" pooling (sampling --> to reduce overfitting by reducing features or selecting most prominent features in an image) --> common types: max or average pooling --> No hyperparameters to learn for gradient descent if chosen since it is usually applied after a convolutional layer (which might/might not include a relu activation function for the final output --> we are sampling from the output of conv layer for the next layer of convolutional layer or final output (softmax regression output)
% 4) Apply and activation function to the output (Relu types such Relu,Leaky Relu are the most popular types to apply nowadays (year: 2020))
% 5) Choose to "apply/do not apply" Fully Connected Networks

% Note: During padding --> Valid = No padding, Same = Pad so that the output size after convolution is the same as the input size
% Note: If stride is used it can act similar to pooling therefore generally in images it is not common to see them together (Other people's opinions so far). Andrew Ng uses stride = 2 and max pooling together in an example in one of his videos
% NOTE: When convolutional layers application is over and and output with 3 dimensions such as MXNXNC is the result of this, before applying the final activation function, the image is flattened into neurons (number of units = MXNXNC) units using all the dimensions (features) and then there is an option to apply fully connected networks which is pretty much feed-forward neural networks with as many layers\units as desired before applying the softmax regression to get the output.
% Note: General formula for finding the output size = MXNXNC (RGB image)
% Note: Filters in conv layers are treated as weights that needs to be learned along with weights at FFNN (fully connected networks) therefore an optimization algorithm such as gradient descent, adam, etc. is required.

% Note: Types of CNN --> 1-D conv, 2-D conv, 3-D conv, 4-D conv:

% 1) For RGB or grayscale images use 2-D conv.
% 2) For language, speech,time-series with 1D vectors use 1-D conv.
% 3) For a video use 3-D conv
% 4) For dynamic systems with spatio-temporal databases use 4-D conv (in general this might change to 3-D conv depending on the problem)

% Example CNN structure: Input --> Conv --> Pool --> Conv --> Pool --> 3 dimensional output --> Flatten --> FCNN --> FCNN --> Softmax --> output (Conv --> Pool is 1 layer together) --> more details in the code


%% GOAL: Train a CNN to recognize C. elegans(worms) with the given training and test data
%% Clean and clear all
clear;
clc;
close all;

%% Import Data

% To generate the same random number each time
rng(100,'twister');

% Trainig data_worm - Loaded and reshaped into a 4-D database
filepath1 = strcat(pwd,'\TrainingData\','Training_worm.mat');

Training_worm = load(filepath1);
training_worm = Training_worm.original_stored;

train_images_worm = reshape(training_worm',[101,101,1,5500]);
train_labels_worm = ones(5500,1);

% Test data_worm - Loaded and reshaped into a 4-D database
filepath2 = strcat(pwd,'\TrainingData\','Test_worm.mat');

Test_worm = load(filepath2);
test_worm = Test_worm.original_stored;

test_images_worm = reshape(test_worm' ,[101,101,1,1500]);
test_labels_worm = ones(1500,1);

% Trainig data_noworm - Loaded and reshaped into a 4-D database
filepath3 = strcat(pwd,'\TrainingData\','Training_noworm.mat');

Training_noworm = load(filepath3);
training_noworm = Training_noworm.original_stored;

train_images_noworm = reshape(training_noworm',[101,101,1,5500]);
train_labels_noworm = zeros(5500,1);

% Test data_noworm - Loaded and reshaped into a 4-D database
filepath4 = strcat(pwd,'\TrainingData\','Test_noworm.mat');

Test_noworm = load(filepath4);
test_noworm = Test_noworm.original_stored;

test_images_noworm = reshape(test_noworm',[101,101,1,1500]);
test_labels_noworm = zeros(1500,1);

%% Combined Data (use this)

Training_images = cat(4,train_images_worm(:,:,:,1:5500),train_images_noworm(:,:,:,1:5500));
Training_labels = [train_labels_worm(1:5500,:);train_labels_noworm(1:5500,:)]; % 1:4125

Test_images = cat(4,test_images_worm,test_images_noworm);
Test_labels =  [test_labels_worm;test_labels_noworm];

validation_images = cat(4,train_images_worm(:,:,:,4126:5500),train_images_noworm(:,:,:,4126:5500));
validation_labels = categorical([train_labels_worm(4126:5500,:);train_labels_noworm(4126:5500,:)]);

%% Randperm Test

% Suffle images & labels randomly
% Shuffle_training_images = randperm(size(Training_images,4));
% Shuffle_test_images = randperm(size(Test_images,4));
% Shuffle_validation_images = randperm(size(validation_images,4));
% 
% Training_images = Training_images(:,:,:,Shuffle_training_images);
% Training_labels = Training_labels(Shuffle_training_images ,:);
% 
% Test_images = Test_images(:,:,:,Shuffle_test_images);
% Test_labels = Test_labels(Shuffle_test_images,:);
% 
% validation_images(:,:,:,Shuffle_validation_images);
% validation_labels = validation_labels(Shuffle_validation_images,:);


% %% Test
% clear;
% clc;
% close all;
% 
% Vector1 = randi(9, 1, 10);
% Vector2 = Vector1;
% ix = randperm(10);
% ShuffeledVector1 = Vector1(ix);
% ShuffeledVector2 = Vector2(ix);
% Q1 = Vector1 == Vector2;                        % Equal
% Q2 = ShuffeledVector1 == ShuffeledVector2;      % Equal

% %% My own test
% clear;
% clc;
% close all;
% 
% rng(100,'twister');
% 
% A = [1,2,3;4,5,6];
% B = [1,1,1;2,2,2];
% 
% shuffling = randperm(size(A,1));
% C = A(shuffling,:);
% D = B(shuffling,:);
% 
% E = rand(16,4);
% F = reshape(E,[4,4,1,4]);
% shuffling4D = randperm(size(F,4));
% G = F(:,:,:,shuffling4D);

%% Generate the CNN layers


%%% INPUT LAYER %%%

% Size of each image is 101X101X1
inputlayer = imageInputLayer([101 101 1],'DataAugmentation','none',...
    'Normalization','zscore','Name','input');



%%% LAYER-1 %%%

% convolution2dLayer(filtersize,numFilters,Name,value) --> additionaly stride, padding, etc. can be added and biases initialized
% BiasLearnRateFactor,etc. multiplies global learning rate with the specified value to change the learning rate for biases,ec.
convlayer1 = convolution2dLayer(5,16,'Stride',2,'Padding',0, ...
    'BiasLearnRateFactor',1,'NumChannels',1,...
    'WeightLearnRateFactor',1, 'WeightL2Factor',1,...
    'BiasL2Factor',1,'BiasInitializer','ones','Name','conv1');

% c.elegans output --> 49*49*16 from convlayer1

% Weights for convlayer1
convlayer1.Weights = rand([5 5 1 16]); % 5 and 16 were the filtersize and numFilters respectively in convlayer1
convlayer1.Bias = rand([1 1 16]);

batchnormlayer1 = batchNormalizationLayer('Name','batchNorm1'); % Batch normalization actually has 2 learnable parameters just like weights to adjust the mean and the variance it uses to normalize the data so that it can normalize the data for better results

relulayer1 = reluLayer('Name','relu1'); % Passing it through a Relu layer (for non-linearity)

% localnormlayer1 = crossChannelNormalizationLayer(3,'Name',...localnorm1','Alpha',0.0001,'Beta',0.75,'K',2); --> Another option that could have been used instead of batchnormalization (details unknown)

maxpoollayer1 = maxPooling2dLayer(4,'Stride',3,'Name','maxpool1','Padding',0);


% c.elegans output --> 16*16*16 from maxpoollayer1

% droplayer1 = dropoutLayer(0.35); --> can be used for randomly dropping some of the connections (as far as I understood)



%%% LAYER-2 %%%

convlayer2 = convolution2dLayer(4,32,'Stride',1, 'Padding',0,...
    'BiasLearnRateFactor',1,'NumChannels',16,...
    'WeightLearnRateFactor',1, 'WeightL2Factor',1,...
    'BiasL2Factor',1,'BiasInitializer','ones','Name','conv2');

% c.elegans output --> 13*13*32 from convlayer2

convlayer2.Weights = rand([4 4 16 32]);
convlayer2.Bias = rand([1 1 32]);

batchnormlayer2 = batchNormalizationLayer('Name','batchNorm2');

relulayer2 = reluLayer('Name','relu2');

% localnormlayer2 = crossChannelNormalizationLayer(3,'Name',...'localnorm2','Alpha',0.0001,'Beta',0.75,'K',2);

maxpoollayer2 = maxPooling2dLayer(3,'Stride',2,'Name','maxpool2','Padding',0);

% c.elegans output --> 6*6*32 from maxpoolayer2

% droplayer2 = dropoutLayer(0.25);



%%% LAYER-3 %%%

% These are fully connected layers (There was no need to flatten the final features which is 6*6*32 = 1152 because the weights already take it into account
% However, there is also a flatten layer function available in matlab which was not used in this case

fullconnectlayer1 = fullyConnectedLayer(5000,'WeightLearnRateFactor',1,...
    'BiasLearnRateFactor',1,'BiasInitializer','ones','WeightL2Factor',1,'BiasL2Factor',1,...
    'Name','fullconnect1');

fullconnectlayer1.Weights = rand([5000 1152]); % 5000 is the output from fullyconnectedlayer1 layer to the next fullyconnectedlayer2
fullconnectlayer1.Bias = rand([5000 1]);

batchnormlayer3 = batchNormalizationLayer('Name','batchNorm3');

relulayer3 = reluLayer('Name','relu3');



%%% LAYER-4 %%%

fullconnectlayer2 = fullyConnectedLayer(2,'WeightLearnRateFactor',1,...
    'BiasLearnRateFactor',1,'BiasInitializer','ones','WeightL2Factor',1,'BiasL2Factor',1,...
    'Name','fullconnect2');

fullconnectlayer2.Weights = rand([2 5000]); % 5000 inputs and 2 outputs for softmax layer due to 2 classes (worm + noworm)
fullconnectlayer2.Bias = rand([2 1]);

batchnormlayer4 = batchNormalizationLayer('Name','batchNorm4');



%%% OUTPUT LAYER %%%

smlayer = softmaxLayer('Name','sml1');

coutputlayer = classificationLayer('Name','coutput');


%% Training options for CNN

% Training options to define hyperparameters, optimizer, and additional
% options such as GPU/CPU usage, plots, and etc. 
% 'Validation patience' can only be used when validation data is available and it is used to stop the Epoch when the results are consistenly similar to each other in validation loss
options = trainingOptions('rmsprop',...
    'LearnRateSchedule','none',...
    'LearnRateDropFactor',0.99,...
    'LearnRateDropPeriod',4,'SquaredGradientDecayFactor',0.99,'ValidationFrequency',60,'ValidationPatience',100,'ValidationData',{validation_images,validation_labels},'L2Regularization',0,... % L2Regularization = 0.0001
    'MaxEpochs',50,'Shuffle','once','Plots','training-progress','ExecutionEnvironment','multi-gpu',...
    'MiniBatchSize',50,'Verbose',1,...
    'CheckpointPath','E:\Classes\ML TTU\PROJ6\checkpoints','InitialLearnRate',0.5); % optimizer = 'sgdm',InitialLearnRate = 0.5, 'Momentum' = ,0.99

%% Layer order for CNN

% Specify the structure of CNN layers during training (layers used in trainNetwork)
layers =[inputlayer, convlayer1,batchnormlayer1, relulayer1, ...
    maxpoollayer1,...
    convlayer2,batchnormlayer2, relulayer2,maxpoollayer2,...
    fullconnectlayer1,batchnormlayer3,relulayer3,fullconnectlayer2,batchnormlayer4, smlayer, coutputlayer];

%% Train the CNN

% store different results coming from different shuffling of data (when shuffling using the built in function, random shuffling can't be controlled --> as far as I know)
store_results = zeros(20,1);

for i=1:1 % can change to match the size of the store_results
    
    % Train the network
    train_im=Training_images;
    train_lb=categorical(Training_labels);
    net.trainParam.goal = 1*10^-3; 
    trainedNet = trainNetwork(train_im,train_lb,layers,options);
    
    % tic-toc to record testing time for all 3000 images
    tic;
    [Ypred,scores] = classify(trainedNet,Test_images);
    score = sum((Ypred==categorical(Test_labels)))/numel(Test_labels);
    store_results(i,1) = score;
    toc;
    
    % Display all the test accuracy obtained (20 of them)
   
    
end


for j = 1:i
fprintf('\nExperiment# %d | Test accucary: %d\n',j,store_results(j,1));
end

%% Evaluate Accuracy on data
clc;
% Training Accuracy
[Ypred,scores] = classify(trainedNet,Training_images);
score = sum((Ypred==categorical(Training_labels)))/numel(Training_labels);
fprintf('\nTraining accuracy: %d\n',score);
% Validation Accuracy
[Ypred,scores] = classify(trainedNet,validation_images);
score = sum((Ypred==categorical(validation_labels)))/numel(validation_labels);
fprintf('\nValidation accuracy: %d\n',score);
% Test Accuracy
[Ypred,scores] = classify(trainedNet,Test_images);
score = sum((Ypred==categorical(Test_labels)))/numel(Test_labels);
fprintf('\nTest accuracy: %d\n',score);

%% Show some example predictions for test

% Combine worm and no worm
test_img = {test_images_worm,test_images_noworm};

% Choose either worm or no worm image to display test results 
j = 1; % Please choose j to be "1" for worm and "2" for no worm images
myimg = test_img{1,j};

for i = 5:10
    current_img = myimg(:,:,1,i);
    [Ypred,scores] = classify(trainedNet,current_img);
    
    imshow(current_img)
    if Ypred == categorical(1)
        title_text = 'Worm';
    else
        title_text = 'No Worm';
    end
    
    title(title_text)
    pause;
    close all;

end

%% Visualize Model Architecture

lgraph = layerGraph(layers);

figure
plot(lgraph);
pause;
close all;