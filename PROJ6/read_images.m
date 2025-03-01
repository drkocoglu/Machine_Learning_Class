function read_images(location,image_name,image_type,Mat_file_name)

% Clear and close all
% clear;
% clc;
% close all;

%% Create variables to save the images for training

% Choose scale to resize the image

% scale = 0.15; % can be adjusted as desired

% Extract image size info to store all flattened images


% image_name = 'image_';
% image_type = '.png';
% location = 'C:\Users\ykocoglu\Desktop\ML TTU\PROJ6\C_elegans\celegans_final\celegans\1\training';

filepath = strcat(location,'\',image_name,num2str(1),image_type);
info = imfinfo(filepath);
Height = info.Height;

% size of original flattened image
flattened_size = Height*Height;

% size of resized flattened image
% resized_height = ceil(Height*scale);
% resized_flattened_size = resized_height*resized_height;

% Create variables to store all flattened images
original_stored = zeros(1,flattened_size);
% original_normalized_stored = zeros(1,flattened_size);

% resized = zeros(1,resized_flattened_size);
% resized_normalized = zeros(1,resized_flattened_size);

% Count number of images in the BoxedImages
a = strcat('\*',image_type) ;
number_of_images = numel(dir(fullfile(location,a)));

% Throw an error if there are no images inside the BoxedImages folder
if number_of_images == 0
    error('There are no images inside the specified folder!\n')
end

%% Import the images
tic;
for i = 1:number_of_images
filepath = strcat(location,'\',image_name,num2str(i),image_type);

% Original Data
original = imread(filepath);
original_converted = im2double(original); % Normalized (original size)
original_reshaped = reshape(original_converted,[1,flattened_size]);

% % Normalize and reshape the image (flatten the image)
% [X_norm_original, ~, ~] = featureNormalize(original_converted);
% X_norm_original_reshaped = reshape(X_norm_original, [1,flattened_size]);
% 
% Store the reshaped images
original_stored(i,:) = original_reshaped;
% original_normalized_stored(i,:) = X_norm_original_reshaped;
% 
% % Resize original images
% resized_image = imresize(original_converted,scale);
% resized_image_reshaped = reshape(resized_image,[1,resized_flattened_size]);
% 
% % Normalize and reshape the resized image (flatten the image)
% [X_norm_resized, ~, ~] = featureNormalize(resized_image);
% X_norm_resized_reshaped = reshape(X_norm_resized, [1,resized_flattened_size]);
% 
% % Store the reshaped & resized images
% resized(i,:) = resized_image_reshaped;
% resized_normalized(i,:) = X_norm_resized_reshaped;
end
toc;

%% Save flattened images as a mat file for later use (Training and Test Data combined)

% Check if TrainingData folder exists in the current directory and if not create it
if ~exist('TrainingData', 'dir')
mkdir TrainingData;
end
% Mat_file_name = 'Training.mat';
filepath = strcat(pwd,'\TrainingData\',Mat_file_name);

save(filepath,'original_stored');
end