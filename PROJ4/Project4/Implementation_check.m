clear;
clc;
close all;

% Load fisheriris data
load fisheriris
% The column vector, species, consists of iris flowers of three different species, setosa, versicolor, virginica. The double matrix meas consists of four types of measurements on the flowers, the length and width of sepals and petals in centimeters, respectively.

% Define the nominal response variable using a categorical array.
sp = categorical(species);
spp = double(sp);
% Fit a multinomial regression model to predict the species using the measurements.  
[B,dev,stats] = mnrfit(meas,spp);

pihat = mnrval(B, meas);

[val,idx]= max(pihat,[],2);

error = nnz(spp - idx);

Error = (error/150)*100;
Accuracy_mnrfit = 100 - Error;


%% Import Data (To check implementation)

%% Set the hyperparameters
number_of_classes = 3;
rho = 0.01; %0.00003;
Beta = 0.99; %0.99;
Batch_number  = 1; % Automatically set (change desired batch size and percentage for training data above)
tolerance = 1*10^-2; % 1*10^-6;
max_iter = 100000;
lambda = 0;
seed = 100;

new_spp = spp - 1; % 0 is required to be the first class in the softmaxregression function due to creation of one-hot-encoding labels within the function

%% Traing and Test using softmaxregression

% Use the softmaxregression function (self-created) to calculate accuracy of training and test data
[Accuracy_training,~,weights] = softmaxregression(meas,meas,new_spp ,new_spp ,number_of_classes,rho,Beta,Batch_number,tolerance,max_iter,lambda,seed);

if Accuracy_mnrfit == Accuracy_training
    fprintf('\nExact Match! Implementation is CORRECT!!!\n')
end