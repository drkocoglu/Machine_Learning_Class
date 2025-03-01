function [Accuracy_training,Accuracy_test,w2] = softmaxregression(training_images,test_images,training_labels,test_labels,number_of_classes,rho,Beta,Batch_number,tolerance,max_iter,lambda,seed)
% This function returns the training and test accuracy using softmax regression.
% One-hot-enconding method is used for labels. 1st label = 0 
% (Make sure first class label ~= 1).
% This function can be used for Batch or Mini-Batch Gradient Descent, with
% or wihtout momentum, and with or without regularization. Any combination
% of the mentioned methods is also possible.
% For Batch Gradient Descent use: Batch_number = 1
% For Mini Batch Gradient Descent use: Batch_number > 1
% For using regularization use: Lambda > 0
% For using momentum use: Beta > 0 where Beta --> [0,1]
% For default seed type: 'Default' otherwise enter any number >= 0


%% Start Training

if Beta == 0
    fprintf('\nWARNING! - Beta = 0 is the same as using NO MOMENTUM in GD updates, this may increase computational time!\n');
elseif lambda == 0
    fprintf('\nWARNING! - lambda = 0 is the same as using NO REGULARIZATION in GD updates, this may cause overfitting!\n');
elseif Batch_number == 1
    fprintf('\nWARNING! - Batch_number = 1 is the same as using the WHOLE BATCH in GD updates, this may slow the convergence time!\n');
end

fprintf('Press Enter to Continue!');
pause;

% Add Bias to Training images
training_images = [training_images,ones(size(training_images,1),1)];

% Add Bias to Training images
test_images = [test_images,ones(size(test_images,1),1)];

%% Create standard labels for 10 classes (One-Hot-Encoding)
classes = eye(number_of_classes,number_of_classes);


%% Create new labels from classes for training and test labels (One-Hot-Encoding)
new_training_labels = zeros(size(training_images,1),number_of_classes);

jj = 0; % Counter for picking the right label for new labels

for iii = 0:(number_of_classes-1) % Go through each class and create new labels for training and test data
    
    index2 = find(training_labels == iii); % Find indices of each class for training
    training_number = numel(index2); % Find the number of elements for each class in training
    
    jj = jj + 1; % Update counter
    
    % New training labels
    for kk = 1:training_number
        location2 = index2(kk,1); % Find the location of new labels
        new_training_labels(location2,:) = classes(jj,:); % Assign new labels
    end
    
end

%% Functions

% Softmax function
y = @(w,x) exp(x*w)./sum(exp(x*w),2);
% Gradient function with regularizaiton (if lambda = 0, no regularization)
Gradient = @(y,t,x,w_prev,lambda) x'*(y-t) + lambda.*w_prev;
% Mini-Batch Gradient Descent with Momentum
V = @(Beta,V,Gradient) Beta.*(V) + (1-Beta).*Gradient; % Velocity function (momentum)
W = @(w,rho,V) w - rho.*V; % Weight update using momentum
% Error Function with regularization (if lambda = 0, no regularization)
Error = @(t,y,w_next,lambda) sum(sum(-1.*t.*log(y),1),2) + (lambda/2).*sum(sum(w_next,1),2);

%% Initial guess for weights and velocity

% Initial w (guess)
rng(seed);
w_initial = rand(size(training_images,2),number_of_classes);
% Initial V (velocity term)
V_initial = rand(size(training_images,2),number_of_classes);

%% Hyperparameters
Batch_size = size(training_images,1)/Batch_number;
%% Gradient Decsent stopping criteria

% Initial graident norm (to enter while loop)
gradient_norm = 1;
% Iteration counter
niter = 0;

%% Gradient Descent re-assigment of initial values
w2 = w_initial;
V2 = V_initial;
Total_Error = zeros(niter,1);
%% Gradient Descent with momentum
tic;

while gradient_norm > tolerance
    % Keep track of number of Epochs
    niter = niter + 1;
    Updated_Batch_size = 0;
    
    
    w2_old = w2; % old w2
    
    for i = 1:Batch_number % Update w at each batch
        
        % Initialize batch
        Initial_Batch_size = Updated_Batch_size + 1;
        Updated_Batch_size = Updated_Batch_size + Batch_size; % 32
        Mini_Batch_training_images = training_images(Initial_Batch_size:Updated_Batch_size,:);
        Mini_Batch_new_training_labels = new_training_labels(Initial_Batch_size:Updated_Batch_size,:);
        
        
        
        % Gradient Descent update
        
        % y = @(w,x) exp(x*w)./sum(exp(x*w),2);
        new_y = y(w2,Mini_Batch_training_images);
        % Gradient = @(y,t,x) x'*(y-t);
        gradient = Gradient(new_y,Mini_Batch_new_training_labels,Mini_Batch_training_images,w2,lambda);
        % V = @(Beta,V,Gradient) Beta.*(V) + (1-Beta).*Gradient;
        V2 = V(Beta, V2, gradient);
        % W = @(w,rho,V) w - rho.*V;
        w2 = W(w2,rho,V2);
        
        
        
    end % End of while loop (gradient descent algorithm ends here)

        % PLOT TRAINING ERROR AT EACH EPOCH
        
        % y = @(w,x) exp(x*w)./sum(exp(x*w),2);
        new_y = y(w2,training_images);
        % Error = @(t,y) sum(sum(-1.*t.*log(y),1),2);
        Total_Error(niter,1) = Error(new_training_labels,new_y,w2,lambda);

   
        % STOPPING CRITERIA (If "gradient" or "w(k+1) - w(k)" <= tolerance or iterations >= max_iter)
        
        
        w2_new = w2; % new w2
        w_norm = norm(w2_new - w2_old); % Break out of while loop if w_norm < tolerance
        % Gradient = @(y,t,x) x'*(y-t);
        gradient = Gradient(new_y,new_training_labels,training_images,w2,lambda);
        gradient_norm = norm(gradient); % Break out of while loop if gradient_norm < tolerance
       
        
        if niter >= max_iter
            fprintf('\nGradient Descent could not converge within max_iter: %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n', max_iter)
            break; % Break out of while loop
            
        elseif any(isnan(w2(:))) % If any of the weights become 'NaN' at any iteration
            fprintf('\nGradient Descent could not converge due to NaN values in w2 @ niter = %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n',niter)
            break; % Break out of while loop
        elseif w_norm < tolerance
            fprintf('\nGradient Descent converged to an answer due to norm(w_new - w_old) < tolerance with w_norm = %d\n',w_norm)
            break;
        end
        
        % PRINT EPOCH#
        fprintf('Epoch | %d\n', niter);
    
end

toc;

%% Plot cost function at the end of the iterations

% Calculate Error with initial w to append to niter = 0
% y = @(w,x) exp(x*w)./sum(exp(x*w),2);
Initial_y = y(w_initial,training_images);
% Error = @(t,y) sum(sum(-1.*t.*log(y),1),2);
Initial_Error = Error(new_training_labels,Initial_y,w_initial,lambda);
% Append Initial Error to the Total Error
Total_Error = [Initial_Error;Total_Error];


figure()
title('Training Error');
xlabel('Epoch#');
ylabel('Training Error');
hold on
plot(0:niter,Total_Error);
legend('Training Error');
hold off

%% Training Error

% Predict new y values for training data and calculate accuracy of training
training_y = y(w2,training_images(1:Updated_Batch_size,:));
[~,idx] = max(training_y,[], 2);
testing = training_labels(1:Updated_Batch_size) - (idx-1);
misclassified_training = nnz(testing);
Training_Error_percent = (misclassified_training/size(training_images,1)).*100;
Accuracy_training = 100 - Training_Error_percent;

fprintf('\nTraining Accuracy = %0.2f\n',Accuracy_training);

%% Testing Error

% Predict new y values for test data and calculate accuracy of testing
test_y = y(w2,test_images(1:size(test_images,1),:));
[~,idx2] = max(test_y,[], 2);
testing2 = test_labels(1:size(test_images,1)) - (idx2-1);
misclassified_test = nnz(testing2);
Test_Error_percent = (misclassified_test/size(test_images,1)).*100;
Accuracy_test = 100 - Test_Error_percent;

fprintf('Test Accuracy = %0.2f\n',Accuracy_test);
end