%% Run this to create the mat files to import images into your CNN application

clear;
clc;
close all;



image_type = '.png';
image_name = 'image_';

location1 = 'C:\Users\ykocoglu\Desktop\ML TTU\PROJ6\C_elegans\celegans_final\celegans\1\training';
location2 = 'C:\Users\ykocoglu\Desktop\ML TTU\PROJ6\C_elegans\celegans_final\celegans\0\training';
location3 = 'C:\Users\ykocoglu\Desktop\ML TTU\PROJ6\C_elegans\celegans_final\celegans\1\test';
location4 = 'C:\Users\ykocoglu\Desktop\ML TTU\PROJ6\C_elegans\celegans_final\celegans\0\test';

Mat_file_name1 = 'Training_worm.mat';
Mat_file_name2 = 'Training_noworm.mat';
Mat_file_name3 = 'Test_worm.mat';
Mat_file_name4 = 'Test_noworm.mat';

read_images(location1,image_name,image_type,Mat_file_name1);
read_images(location2,image_name,image_type,Mat_file_name2);
read_images(location3,image_name,image_type,Mat_file_name3);
read_images(location4,image_name,image_type,Mat_file_name4);