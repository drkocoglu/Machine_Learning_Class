clc;
clear;
close all;

% Find the number of images inside the NemaLife Images_Converted folder (change the folder name if renamed)
a = dir(fullfile('./NemaLife Images_Converted','/*.jpg'));
number_of_images = numel(a);
%%
% Delete the contents of BoxedImages folder to store new images generated
% delete('BoxedImages/*');

% Create new variables to store images in the for loop below
%ima = cell(1,151);
%maxi = zeros(1,151);
%cImg=struct('Images',zeros(10,1), 'Std',zeros(10,3));

% Box# counter to track box# coming from different sources of images
k = 1;
L = 0;


% Remove boxes with very small and large size (not worm) and store the bboxes of all the images

%%
% Read each image
this_img = imread(sprintf('./NemaLife Images_Converted/%05d.jpg',8));
%%
[bboxes, images] = DetectionHelper_function(this_img);
images=rgb2gray(images);
%%
n = 2;

% Calculate the area of each rectangle box
area = bboxes(:,3) .* bboxes(:,4);
% Find a threshold for very large and very small boxes
thre = [round(mean(area) + 10000), round(mean(area)+800)];
% Find the required size for a square box
x = find(area <= thre(1) & area >= thre(2));
maxi(n)  = max(area(x));
m = max(maxi);
sd = floor(sqrt(m));
% Pick only the boxes which does not satisfy the threshold and change rectangles to a square of same size 
bx1 = bboxes(x,:);
bx1(:,3:4) = sd;

% Place the object in the center of the image
box_original = bboxes(x,:);
diff = bx1(:,3:4) - box_original(:,3:4);
bx1(:,1:1) = box_original(:,1:1) - (diff(:,1:1)/2);
bx1(:,2:2) = box_original(:,2:2) - (diff(:,2:2)/2);

% Track box# of each image continously
k = L +1;
L = L + length(bx1);

% Save cropped boxes from each image into a structure along with its std, row size, and column size
% for i = k : L
%fprintf(k)
%fprintf(L)
imgs = insertShape(this_img,'Rectangle',bx1(:,:),'LineWidth',5, color='r');
%imgs = rgb2gray(imgs);
imshow(imgs)
    %hold on;
% end

%hold off;

%imshow(imgs)