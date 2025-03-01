% Assignment#2 - Machine Learning
% Name: Yildirim Kocoglu

% Clear and close all
clear;
clc;
close all;

% Counter for subplots
j = 0;
k = 0;

% OPTION FOR USING GRADIENT DESCENT!!! (BONUS!)

option ='on'; % Plese choose 'on' to use gradient descent for COMPARISON purposes only! (Analytical solution is used by default)

% If statement to give early warnings depending on the option choice
if strcmp(option,'on')
    fprintf('Gradient Descent solution is USED!\n');
elseif strcmp(option,'off')
    fprintf('Gradient Descent solution is NOT USED!\n');
else
    error('Please choose "on" or "off" to turn on/off plot. CHECK THE SPELLING!');
end

fprintf('E_RMS vs M plot is from ANALYTICAL SOLUTION (DEFAULT)!\n\n')
fprintf('Paused!!!\nPlease press any button after clicking on "Command Window" to continue!\n\n');
pause; % Paused for seeing early warnings!


% For loop for generating N = 10 & N = 100 training samples
for N = [10,100]
    
    % Generation of training data "x_train" & training target "t_train"
    rng(55,'twister'); %55th seed (keeps "random" selection the same)
    
    x_train = rand(N,1); % Uniform random sampling between 0-1
    x_train = sort(x_train, 'descend'); % helps with plotting
    epsi_train = normrnd(0,0.3,[N,1]); % Noise from Gaussian (Normal) distribution N(mu=0, std = 0.3)
    t_train = sin(2*pi*x_train) + epsi_train; % Target function with noise added
    x_train_norm = (x_train - mean(x_train))/std(x_train); % Normalizing the data (1st attempt)
    
    % Generation of test data x_test & test target t_test
    x_test = rand(100,1);
    x_test = sort(x_test, 'descend'); % helps with plotting
    spaced_var = linspace(0,1,300)'; % Linspace for more datapoints (Helps with plotting,comparing, and testing)
    spaced_var_norm = (spaced_var - mean(x_train))/std(x_train); % Normalizing spaced variable
    epsi_test = normrnd(0,0.3,[100,1]); % Noise from Gaussian (Normal) distribution N(mu=0, std = 0.3)
    t_test = sin(2*pi*x_test) + epsi_test; % Target function with noise added
    x_test_norm = (x_test - mean(x_train))/std(x_train); % Normalizing the data (using training data mean and std)
    
    % Desired degree of fit (0-M)
    M = 9;
    
    
    %legend([p1,p2,p3],'Training Data', 'Test Data', sprintf('Polynomial fit (Matlab function) degree: %d', M))
    %hold off
    
    
    
    % Set max_iter, lambda, and initialize J_train and J_test
    max_iter = 10000000;
    lambda = 0.5; % Use for regularization (lambda = 0: no regularization) - CAN BE ADJUSTED!
    
    % Initialize training & test error
    JJ_train = zeros(M+1,1);
    JJ_test = zeros(M+1,1);
    
    
    % Change degree of polynomial in a for loop (0-9)
    for MM = 0:M
        f2 = ones(size(x_test,1),MM+1); % Initialize size for polynomial features (test data)
        f3_analytical = ones(size(x_train,1),MM+1); % Used for checking analytical solution!!!
        f4_spaced_var = ones(size(spaced_var,1),MM+1);
        f5_norm_test = ones(size(x_train,1),MM);
        
        % Creating desired polynomial in a for loop
        for i =1:MM
            f2(1:end,i+1) = x_test_norm.^i; % test data
            f3_analytical(1:end,i+1) = x_train_norm.^i; % training data (analytical training data)
            f4_spaced_var(1:end,i+1) = spaced_var_norm.^i; % test data (analytical test data)
            f5_norm_test(1:end,i) = x_train.^i; % training data
        end
        f5_norm_test = (f5_norm_test - mean(f5_norm_test))./std(f5_norm_test);
        f5_norm_test = [ones(size(f5_norm_test,1),1), f5_norm_test];
        
        % GRADIENT DESCENT ALGORITHM!
        
        % Set initial gradient_norm and tolerance in the following scale:
        % gradient_norm > tolerance (To avoid issues within the while loop) 
        
        rho = 0.001; % Learning rate (CAN BE ADJUSTED!):
        w2_norm_test = rand(MM+1,1); % random (0-1) initial guess for w
        gradient_norm = 1; % Initial gradient_norm to enter while loop
        tolerance = 1*10^-1; % Desired gradient norm tolerance to exit while loop for Gradient descent (CAN BE ADJUSTED!)
        niter = 0; % iteration counter
        
        
        
        
        % If statement to control whether to use or not use gradient descent!
        if strcmp(option,'on')
            
            tic; % Keep track of elapsed time (Gradient descent convergence)
            while gradient_norm > tolerance
                
                
                gradient_norm_test = (2.*w2_norm_test'*(f5_norm_test'*f5_norm_test) - 2.*t_train'*f5_norm_test) + 2*lambda*w2_norm_test';
                w2_norm_test = w2_norm_test - rho.*gradient_norm_test';
                y_norm_test = f5_norm_test*w2_norm_test;
                
                % Keep track of number of iterations
                niter = niter + 1;
                
                % STOPPING CRITERIA (If gradient <= tolerance or iterations >= 1000000)
                
                % gradient_norm = norm(gradient); % Break out of while loop if gradient_norm < tolerance
                
                % gradient_norm_test
                gradient_norm = norm(gradient_norm_test);
                if niter >= max_iter
                    fprintf('Gradient Descent could not converge within max_iter: %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n', max_iter)
                    break; % Break out of while loop
                    
                elseif any(isnan(w2_norm_test)) % If any of the weights become 'NaN' at any iteration
                    fprintf('Gradient Descent could not converge due to NaN values in w2 @ niter = %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n',niter)
                    break; % Break out of while loop
                end
                
                
            end % End of while loop (gradient descent algorithm ends here)
            
            toc; % Keep track of elapsed time
            % Keep track of iteration# to converge for each degree of polynomial experiment (M = 0,...,9)
            fprintf('M = %d is complete\nNumber of iterations: %d\n\n', MM, niter);
            
            % elseif strcmp(option,'on')
            %fprintf('Gradient Descent is not used!\n');
            
            % Analytical solution in the loop
            w = f3_analytical\t_train;
            y11 = f3_analytical*w; %training
            y12 = f2*w; % test
            y_spaced_var = f4_spaced_var*w; % test (more datapoints for better plots)
            
            % Calculating training and test J(w) at each iteration
            J_analytical(1,1) = sum((t_train - y11).^2);
            J_testing(1,1) = sum((t_test - y12).^2);
            
            % Store E_RMS for plotting
            JJ_train(MM+1,1) = sqrt(J_analytical(1,1)/length(x_train));
            JJ_test(MM+1,1) = sqrt(J_testing(1,1)/length(x_test));
            
        elseif strcmp(option,'off')
            % Analytical solution in the loop
            w = f3_analytical\t_train;
            y11 = f3_analytical*w; %training
            y12 = f2*w; % test
            y_spaced_var = f4_spaced_var*w; % test (more datapoints for better plots)
            
            % Calculating training and test J(w) at each iteration
            J_analytical(1,1) = sum((t_train - y11).^2);
            J_testing(1,1) = sum((t_test - y12).^2);
            
            % Store E_RMS for plotting
            JJ_train(MM+1,1) = sqrt(J_analytical(1,1)/length(x_train));
            JJ_test(MM+1,1) = sqrt(J_testing(1,1)/length(x_test));
        else
            error('Please choose "on" or "off" to turn on/off plot. CHECK THE SPELLING!');
        end
        
    end % End of for loop MM = 0:M
    
    fprintf('N = %d is complete\n\n\n',N);
    
    
    % Plot last polynomial degree for analysis (degree = 9)
    
    k = k + 1; % counter for subplot
    
    figure(1)
    subplot(2,2,k);
    hold on
    
    p1 = scatter(x_train,t_train,50,'x','r','LineWidth',2);
    p2 = scatter(x_test,t_test,50,'filled','d','g','LineWidth',2);
    title(sprintf('POLYFIT SOLUTION (N = %d, DEGREE: %d)',N,M));
    xlabel('x');
    ylabel('t');
    
    % (Matlab's Polyfit function for fitting nth degree polynomial)
    f_x = polyfit(x_train,t_train,M);
    x1 = 0:0.001:1;
    y1 = polyval(f_x,x1);
    
    p3 = plot(x1,y1','r--');
    legend([p1,p2,p3],'Training Data', 'Test Data',sprintf('Polyfit function solution'));
    ylim([-2 1.5]);
    
    k = k + 1; % counter for subplot
    
    subplot(2,2,k);
    hold on
    title(sprintf('IMPLEMENTED SOLUTION (N = %d, DEGREE: %d)',N,M));
    xlabel('x');
    ylabel('t');
    p4 = scatter(x_train,t_train,100,'x','r','LineWidth',2);
    p5 = scatter(x_test,t_test,50,'filled','d','g','LineWidth',2);
    p9 = plot(x_train, y11,'bo-');
    p13 = plot(spaced_var, y_spaced_var, 'k');
    
    
    if strcmp(option, 'on')
        p_norm_test =  plot(x_train, y_norm_test, '-mo');
        legend([p4,p5,p9,p13,p_norm_test],'Training Data', 'Test Data',sprintf('Analytical solution (training)'),sprintf('Analytical solution (test)'),'Gradient-Descent (training)');
    elseif strcmp(option, 'off')
        
        legend([p4,p5,p9,p13],'Training Data', 'Test Data',sprintf('Analytical solution (training)'),sprintf('Analytical solution (test)'));
    else
        error('Please choose "on" or "off" to turn on/off plot. CHECK THE SPELLING!');
    end
    
    ylim([-2 1.5]) % limit the y-axis
    hold off
    
    
    j = j + 1; % counter for subplot
    
    % Plot E_RMS
    figure (2);
    subplot(1,2,j);
    hold on
    title(sprintf('E_R_M_S vs M (N = %d)',N));
    xlabel('M');
    ylabel('E_R_M_S');
    p11 = plot(0:M,JJ_train(:,1),'bo-');
    p12 = plot(0:M,JJ_test(:,1),'ro-');
    legend([p11,p12], 'Training', 'Test');
    hold off
end