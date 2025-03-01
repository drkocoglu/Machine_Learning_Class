% Assignment#3 - Machine Learning
% Name: Yildirim Kocoglu

% Clear and close all
clear;
clc;
close all;

% Initialize Lambda values in a vector
trial = linspace(-3, 3,50); % Ln(Lambda)
trial = exp(trial); % Actual Lambda

% Setup for number of subplots (4 subplots to show effects of different lambdas on 100 models)
k = round(linspace(1,50,4));
counter = 0;

% Initial samples and models
N = 25; % Number of samples
L = 100; % Number of models

% Generation of RBF terms (RBF_NO is adjustable. Choose a higher value for more RBF terms.

% WARNING!!!: A value higher than RBF_NO = 25 will require a larger range of lambda to fully observe the changes in bias^2, variance, etc. graph...
%... and is not recommended in this case)
RBF_NO = 11; % Desired Number of RBF terms
lower_limit = 0; % lower limit of location
upper_limit = 1; % upper limit of location
new_mu = linspace(lower_limit,upper_limit,RBF_NO);  % Locations of RBF terms
sigma = 0.1; % Constant (given) sigma for each RBF term

RBF = @(X,mu,sigma) exp(-(X-mu).^2/(2*sigma.^2)); % RBF function for each term

% Initialize for storing x_train, t_train,
storage_x_train = zeros(N,L);
storage_t_train = zeros(N,L);
storage_fx = zeros(N,L);
storage_t_test = zeros(1000,L);
storage_w = zeros(RBF_NO+1,L);
storage_t_test_true = zeros(1000,L);

% Initialize Storage for plotting Bias^2, Variance, (Bias^2 + Variance), Test Error
storage_bias = zeros(size(trial,2),1);
storage_variance = zeros(size(trial,2),1);
storage_bias_variance = zeros(size(trial,2),1);
storage_test_error = zeros(size(trial,2),1);

% Use seed for the same samples each time
rng(55,'twister');

% Generate test data between 0-1 (uniformly distrubuted)
test = linspace(0,1,1000)';
h_test = sin(2*pi*test);
t_test = sin(2*pi*test) + normrnd(0,0.3,[1000,1]);

% Generate the training data
x_train = rand(N,L); % Uniform random sampling between 0-1

x_train = sort(x_train, 'descend'); % helps with plotting
epsi_train = normrnd(0,0.3,[N,L]); % Noise from Gaussian (Normal) distribution N(mu=0, std = 0.3)
t_train = sin(2*pi*x_train) + epsi_train; % Target function with noise added

% To plot test data for 100 different models
repeated_matrix = repmat(test,1,100);

% Test RBF function
repeated_x_test = repmat(test,1,RBF_NO); % Training sample x (from current model) repeated RBF_No times
rbf_test =[ones(size(test,1),1), RBF(repeated_x_test,new_mu,sigma)];


% Loop over different lambda values
for j = 1:length(trial)
    
    lambda = trial(1,j);
    
    % Loop over each model (with chosen lambda)
    for i = 1:L
        
        x_train_model = x_train(:,i);
        t_train_model = t_train(:,i);
        
        storage_x_train(:,i) = x_train_model;
        storage_t_train(:,i) = t_train_model;
        
        % Analytical solution
        
        % Training RBF
        repeated_x = repmat(x_train_model,1,RBF_NO); % Training sample x (from current model) repeated RBF_No times
        rbf =[ones(size(x_train_model,1),1), RBF(repeated_x,new_mu,sigma)];
        w = inv(rbf'*rbf + lambda.*eye(size(rbf,2)))*rbf'*t_train_model;
        t = rbf*w;
        storage_fx(:,i) = t;
        storage_w(:,i) = w;
        
        % Test RBF
        y_test = rbf_test*w;
        storage_t_test(:,i) = y_test;
        storage_t_test_true(:,i) = t_test;
        
    end
    
    % Plot all 100 models with different lambda values (4 subplots)
    if any(ismember(k,j))
        counter = counter + 1;
        figure(1)
        subplot(2,2,counter)
        hold on;
        title('x_test vs t_test');
        scatter(test,t_test,50,'o','g','LineWidth',2);
        plot(repeated_matrix,storage_t_test);
        legend('Test data',sprintf('ln{\\lambda} = %d', log(lambda)));
        hold off;
    end
    
    % Calculating Bias^2, Variance, (Bias^2 + Variance), Average Test Error(over 100 models)
    mean_storage_fx = mean(storage_t_test,2);
    bias = mean((h_test-mean_storage_fx).^2);
    variance = mean(mean((storage_t_test - mean_storage_fx).^2,2));
    bias_variance = bias + variance;
    test_error = mean(mean((storage_t_test_true - storage_t_test).^2,2));
    
    % Store all values for different lambda values (Bias^2, Variance, (Bias^2 +Variance), Average Test Error(over 100 models))
    storage_bias(j,:) = bias;
    storage_variance(j,:) = variance;
    storage_bias_variance(j,:) = bias_variance;
    storage_test_error(j,:) = test_error;
    
end

% Plot Bias^2/Variance/(Bias^2 + Variance)/Test Error vs ln(lambda)
figure(2)
hold on;
xlabel('ln({\lambda})');
p1 = plot(log(trial),storage_bias,'bo-');
p2 = plot(log(trial),storage_variance,'ro-');
p3 = plot(log(trial),storage_bias_variance,'mo-');
p4 = plot(log(trial),storage_test_error,'ko-');
legend([p1,p2,p3,p4],'Bias^2','Variance','Bias^2 + Variance','Test error');
hold off;