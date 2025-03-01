%% Project#4 - Machine Learning
% Group Members: Yildirim Kocoglu, Ishtiaque Zaman, Denzel Smith, Vaishnavee Sharma

clear;
clc;
close all;


%% Import Data
tic;

% Load Original.mat (Training & Test data from TrainingData folder)
filepath = strcat(pwd,'\TrainingData\','Original.mat');

Original = load(filepath);
original = Original.original_stored;
original_normalized = Original.original_normalized_stored;
resized = Original.resized;
resized_normalized = Original.resized_normalized;

% Indices that need to be deleted from the Labels (due to the modification to the images)
Index_miss = Original.idx2;

%% Import Labels

Labels = xlsread('C:\Users\ykocoglu\Desktop\ML TTU\PROJ4\Project4\Combined_Labels.xlsx', 1, 'A1:A10900');
Labels(Index_miss) = [];
Labels(Labels == 3) = 1; % Multiple worms is a worm class (modified later)
Labels(Labels == 2) = 1; % Worm + Defect is a worm class (modified later)
% filename = 'Combined_Labels_test.xlsx';
% writematrix(Labels,filename,'Sheet',1,'Range','A1');
toc;

%% Find the number divisible by desired batch and training size (per desired batch)
Total_size = size(original,1);
desired_percent_training = 0.6; % can be adjusted as desired
Training_size = ceil(Total_size*desired_percent_training);
desired_batch_size = 32; % can be adjusted as desired
Remaining_to_remove = rem(Training_size,desired_batch_size);
New_Training_size = Training_size - Remaining_to_remove;
%% Divide the data into training and test

% Divide Labels
training_labels =  Labels(1:New_Training_size,:); % Labels(1:((size(original,1))*0.6),:);
test_labels =  Labels(New_Training_size+1:end,:); % Labels(((size(original,1))*0.6)+1:end,:);

% Divide Images into training and test data

% Original image
original_training = original(1:New_Training_size,:);
original_testing = original(New_Training_size+1:end,:);
% Original image (normalized)
original_training_normalized = original_normalized(New_Training_size+1:end,:);
original_testing_normalized = original_normalized(New_Training_size+1:end,:);
% Resized image
resized_training = resized(1:New_Training_size,:); 
resized_testing =  resized(New_Training_size+1:end,:); 
% Resized image (normalized)
resized_training_normalized =  resized_normalized(1:New_Training_size,:);
resized_testing_normalized =   resized_normalized(New_Training_size+1:end,:); 

%% Set the hyperparameters
number_of_classes = 2;
rho = 0.00003; 
Beta = 0.99; 
Batch_number  = New_Training_size/desired_batch_size; % Automatically set (change desired batch size and percentage for training data above)
tolerance = 1*10^-4; 
max_iter = 5000;
lambda = 0;
seed = 100;

%% Traing and Test using softmaxregression

% Use the softmaxregression function (self-created) to calculate accuracy of training and test data
[Accuracy_training_original,Accuracy_test_original,weights] = softmaxregression(resized_training_normalized,resized_testing_normalized,training_labels,test_labels,number_of_classes,rho,Beta,Batch_number,tolerance,max_iter,lambda,seed);
