% WARNING: The name of the variables for weights and some other terms for calculating the output for
% different parts of the project are the same. The only way to see the individual results of each part is to run the code in sections.
% Otherwise, the results will be updated in the next part (section)...
% This was done to avoid further confusion at the end of the run by creating more variables than necessary in the workspace of matlab.

% WARNING: The script pauses at the end of each part (part 1, part 2.1, part 2.2)for the user convenience before proceeding if, 
% the user chooses to run all of the code at once and observe the final results instead of running the code one section at a time.


% WARNING: RELU DID NOT WORK WITHOUT MOMENTUM IN THE CLASSIFICATION PROBLEM
%% Clear and close all
clear;
clc;
close all;


%% General Functions for Part 1 and Part 2

% Gradient Descent function
w = @(w,rho,gradient) w-rho.*gradient;
% Jacobian for softmax regression and LS regression
J = @(y,t) y-t;
% Gradient Descent with Momentum
V = @(Beta,V,Gradient) Beta.*(V) + (1-Beta).*Gradient; % Velocity function (momentum)
W = @(w,rho,V) w - rho.*V; % Weight update using momentum

%% Part 1 Functions

% sigmoid activation
h_sigmoid = @(a) 1./(1+exp(-a));
% RELU activation function (not used)
h_relu = @(a) max(0,a);
% RELU activation function derivative (not used)
h_relu_derivative = @(a) double(a >= 0); % Find a logical matrix and converts to double matrix with 0 and 1 elements (unlike other gradients this one does not take z_1 it takes a_1 (no h(a) in the equation)
% Softmax function
h_softmax = @(a) exp(a)./sum(exp(a),2);
% Cost Function (softmax regression)
Error_SM = @(t,y) sum(sum(-1.*t.*log(y),1),2);
% Derivative of sigmoid activation function
h_sigmoid_derivative = @(h) h.*(1-h);


%% Part 2 Functions

% Hyperbolic tangent function
h_tanh = @(a) tanh(a);
% Hyperbolic tangent derivative
h_tanh_derivative = @(h) 1-h.^2;
% Cost Function (Least squares regression)
Error_LS = @(t,y) sum((y-t).^2);

%% Part 1 (Classification problem) 

% (One hidden layer composed of 2 units)
% Hyperparameters
rho = 0.01; % 0.01
lambda = 0; % This is not used in this part
Beta = 0.99; % This is not used in this part

% Gradient Descent stopping criteria
% Initial gradient norm to enter while loop (gradient descent)
gradient_norm = 1;
% Tolerance to stop iterations
tolerance = 1*10^-3;
% Maximum iterations to stop calculations
max_iter = 1000000;
% Iteration counter to track number of iterations
niter = 0;

% Data 
C1 = [-1,1;-1,1]';
C2 = [-1,1;1,-1]';
Data = [C1;C2]; % Combined data


% Labels (maually generated)
Labels = [1,0;1,0;0,1;0,1];

% Initialize weights for each layer and bias terms at each layer
rng(100);
% Initial w (from input to hidden layer)
w1 = rand(2,2);
V1 = rand(2,2);
w1_b = rand(1,2); % Bias
V1_b = rand(1,2);
% Initial w (from hidden layer to output)
w2 = rand(2,2);
V2 = rand(2,2);
w2_b = rand(1,2); % Bias
V2_b = rand(1,2);


% Gradient Descent

% Track Cost vs niter
Total_Error = zeros(niter,1);
tic;
while gradient_norm > tolerance
    
    % Forward propogation
    
    % Calculate bias terms
    b_1 = ones(4,1)*w1_b; % 4X2
    b_2 = ones(4,1)*w2_b; % 4X2
    % Calculate the output
    a_1 = Data*w1 + b_1;
    z_1 = h_sigmoid(a_1); % Using sigmoid activation function (input layer)
    a_2 = z_1*w2 + b_2;
    z_2 = h_softmax(a_2); % Using softmax activation function (output layer)
    
    % Total cost
    Cost = Error_SM(Labels,z_2);
    fprintf('Iteration %d | Cost %d\n',niter,Cost);
    % Store cost at each iteration
    Total_Error(niter+1,1) = Cost;
    
    % Backward propogation
    
    % Deltas required for gradient calculation
    d_3 = J(z_2,Labels); % 4X2
    d_2 = d_3*w2'.*h_sigmoid_derivative(z_1); % z_1 otherwise; % Is this correct?
    % Gradients for w1 and w2
    del1 = Data'*d_2; % 2X2 (w1 matrix gradients w.r.t cost)
    del2 = z_1'*d_3; % 2X2 (w2 matrix gradients w.r.t cost)
    % Gradients for offset bias terms w1_b and w2_b
    del1_b1 = sum(d_2,1);
    del2_b2 = sum(d_3,1);
    
%     w1 = w(w1,rho,del1);
%     w2 = w(w2,rho,del2);
%     w1_b = w(w1_b,rho,del1_b1);
%     w2_b = w(w2_b,rho,del2_b2);
    
    V1 = V(Beta,V1,del1);
    w1 = W(w1,rho,V1);
    V2 = V(Beta,V2,del2);
    w2 = W(w2,rho,V2);
    V1_b = V(Beta,V1_b,del1_b1);
    w1_b = W(w1_b,rho,V1_b);
    V2_b = V(Beta,V2_b,del2_b2);
    w2_b = W(w2_b,rho,V2_b);
    
    
    % STOPPING CRITERIA (If "gradient" or "w(k+1) - w(k)" <= tolerance or iterations >= max_iter)
    
    gradient = [del1(:);del2(:);del1_b1(:);del2_b2(:)];
    w_norm = [w1(:);w2(:);w1_b(:);w2_b(:)];
    % Gradient = @(y,t,x,w_prev,lambda) x'*(y-t) + lambda.*w_prev;
    gradient_norm = norm(gradient); % Break out of while loop if gradient_norm < tolerance
    
    % Keep track of number of iterations
    niter = niter + 1;
    
    if niter >= max_iter
        fprintf('\nGradient Descent could not converge within max_iter: %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n', max_iter)
        break; % Break out of while loop
        
    elseif any(isnan(w_norm(:))) % If any of the weights become 'NaN' at any iteration
        fprintf('\nGradient Descent could not converge due to NaN values in w2 @ niter = %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n',niter)
        break; % Break out of while loop
        
    elseif gradient_norm < tolerance
        fprintf('\nGradient Descent converged to an answer due to norm(gradient) < tolerance with w_norm = %d\n',gradient_norm)
        break; % Break out of while loop
        
    end
    

    

end
toc;
% Plot cost vs iter#

figure()
title('Part 1 - Training Error');
xlabel('Epoch#');
ylabel('Training Error');
hold on
plot(1:niter,Total_Error);
legend('Training Error');
hold off


% Plot Decision boundary

% Classes for each examples given C1 = 1, C2 = 0
y = [1;1;0;0];

% Grid range
u = linspace(-2, 2, 100);
v = linspace(-2, 2, 100);

% Create 3-D Decision Boundary
[X,Y] = meshgrid(u,v);
X = reshape(X,[10000,1]);
Y = reshape(Y,[10000,1]);
test = [X,Y];
a1 = test*w1 + ones(10000,1)*w1_b;
z1 = h_sigmoid(a1);
a2 = z1*w2 + ones(10000,1)*w2_b;
z2 = h_softmax(a2);
z2_new = reshape(z2(:,1),[100,100]);


% Alternative method for creating 3-D decision boundary

% Initialize z (final answer) storage
z = zeros(length(u), length(v));

for i = 1:length(u)
    for j = 1:length(v)
        % Calculate bias terms
        b_1 = ones(1,1)*w1_b; % 4X2
        b_2 = ones(1,1)*w2_b; % 4X2
        % Calculate the output
        a_11 = [u(i), v(j)]*w1 + b_1;
        z_11 = h_sigmoid(a_11); % Using sigmoid activation function (input layer)
        a_22 = z_11*w2 + b_2;
        z_22 = h_softmax(a_22);
        z_222 = find(z_22 >= 0.5);% Using softmax activation function (output layer)
        
        if z_222 == 1
            z_2222 = z_22(:,1);
        elseif z_222 == 2
            z_2222 = z_22(:,1);
        end
        z(i,j) = z_2222; % meshrid results
    end
end

% 2-D Decision Boundary
figure()
p1 = scatter(Data(1:2,1),Data(1:2,2),100,'filled','k','s');
hold on
p2 = scatter(Data(3:4,1),Data(3:4,2),100,'filled','r','d');
contour(u, v, z2_new, [0.99, 0.99],'r','LineWidth', 2);
title('Part 1 - 2-D Decision Boundary');
xlabel('X1');
ylabel('X2');
legend([p1,p2],'Class 1','Class 2');
hold off

% 3-D Decision Boundary
figure()
mesh(u,v,z2_new);
title('Part 1 - 3-D Decision Boundary');
xlabel('X1');
ylabel('X2');
zlabel('Prediction');
colormap(jet);
colorbar
caxis([0 1])
hold on
p3 = scatter3(Data(1:2,1),Data(1:2,2),y(1:2,:),100,'filled','k','s');
p4 = scatter3(Data(3:4,1),Data(3:4,2),y(3:4,:),100,'filled','r','d');
legend([p3,p4],'Class 1','Class 2');
hold off


% Gradient Checking 

% Normally, this is done before running the gradient descent algorithm but, removed from the main body of the code due to potential for confusion...

% This is done purely to re-confirm the gradients calculated in the
% back propogation part and otherwise an unnecessary addition to the
% code...

% Unroll the weigths
w1_vector = reshape(w1,[1,4]);
w2_vector = reshape(w2,[1,4]);

% Combine the unrolled weights
w_vector = [w1_vector,w2_vector];

% Store calculated gradients
gradapprox = zeros(1,1);

% Choose a small epsilon for numerical method for calculating gradients
epsilon = 1*10^-6;

% Calculate gradients for w1 and w2 which indirectly confirms whether the user implementation of back propogation is correct or wrong... 
% It will also indirectly confirm the correctness of gradients for bias terms since they are related to the gradients of w1 and w2 directly...
for i = 1:8

thetaplus = w_vector; % Initialize to w vector which contains all the weigths from both hidden and output layers for adding epsilon
thetaplus(i) = thetaplus(i) + epsilon;
thetaplus1 = reshape(thetaplus(:,1:4),[2,2]); % w1_plus
thetaplus2 = reshape(thetaplus(:,5:8),[2,2]); % w2_plus

thetaminus = w_vector; % Initialize to w vector which contains all the weigths from both hidden and output layers for subtracting epsilon
thetaminus(i) = thetaminus(i) - epsilon;
thetaminus1 = reshape(thetaminus(:,1:4),[2,2]); % w1_minus
thetaminus2 = reshape(thetaminus(:,5:8),[2,2]); % w2_minus

a_1_plus = Data*thetaplus1+ b_1; % w1_plus
a_1_minus = Data*thetaminus1 + b_1; % w1_minus
z_1_plus = h_sigmoid(a_1_plus);
z_1_minus = h_sigmoid(a_1_minus);
a_2_plus = z_1_plus*thetaplus2 + b_2; % w2_plus
a_2_minus = z_1_minus*thetaminus2 + b_2; % w2_minus
z_2_plus = h_softmax(a_2_plus);
z_2_minus = h_softmax(a_2_minus);

Cost_eps_plus = Error_SM(Labels,z_2_plus);
Cost_eps_minus = Error_SM(Labels,z_2_minus);
gradapprox(i) = (Cost_eps_plus - Cost_eps_minus)/(2*epsilon);

end

% Calculate difference in numerical and backprop methods of calculating gradients
approx_grad_w1 = reshape(gradapprox(:,1:4),[2,2]);
approx_grad_w2 = reshape(gradapprox(:,5:8),[2,2]);
relative_difference_w1 = norm(approx_grad_w1 - del1)/norm(approx_grad_w1 + del1);
relative_difference_w2 = norm(approx_grad_w2 - del2)/norm(approx_grad_w2 + del2);

% Pause before continuing to the next part
fprintf('\nPart 1 is completed! Please continue for Part 2.1 with 3 hidden units...\n');
pause;

%% Part 2 (Regression problem - 3 hyperbolic tangent hiddent units)

% (One hidden layer composed of 3 hyperbolic tangent hidden units)


% Create Training data 
rng(100) 
X=2*rand(1,50)-1;
X = sort(X, 'descend');
T=sin(2*pi*X)+0.3*randn(1,50);
X = X';
X_norm = (X - mean(X))./std(X);
T = T';

% Create Test data 
X_test=linspace(-1,1,300);
X_test = sort(X_test, 'descend');
T_actual=sin(2*pi*X_test);
X_test = X_test';
X_norm_test = (X_test - mean(X))./std(X);
T_actual = T_actual';

% Hyperparameters
rho = 0.01;
lambda = 0; % This is not used here
Beta = 0.99;

% Gradient Descent stopping criteria
% Initial gradient norm to enter while loop (gradient descent)
gradient_norm = 1;
% Tolerance to stop iterations
tolerance = 1*10^-3;
% Maximum iterations to stop calculations
max_iter = 1000000;
% Iteration counter to track number of iterations
niter = 0;


% Initialize weights for each layer and bias terms at each layer
rng(100);
% Initial w (from input to hidden layer)
w1 = rand(1,3);
V1 = rand(1,3);
w1_b = rand(1,3); % Bias
V1_b = rand(1,3);
% Initial w (from hidden layer to output)
w2 = rand(3,1);
V2 = rand(3,1);
w2_b = rand(1,1); % Bias
V2_b = rand(1,1);


% Gradient Descent

% Track Cost vs niter
Total_Error = zeros(niter,1);
tic;
while gradient_norm > tolerance
    
    
    % Forward propogation
    
    % Calculate bias terms
    b_1 = ones(50,1)*w1_b; % 50X3
    b_2 = ones(50,1)*w2_b; % 50X1
    % Calculate the output
    a_1 = X_norm*w1 + b_1; % 50X3
    z_1 = h_tanh(a_1); % Using tanh activation function (input layer) % 50X3
    a_2 = z_1*w2 + b_2; % 50X1
    z_2 = a_2; % Using h(a) = a (output layer) % 50X1
    
    % Total cost
    Cost = Error_LS(T,z_2);
    fprintf('Iteration %d | Cost %d\n',niter,Cost);
    % Store cost at each iteration
    Total_Error(niter+1,1) = Cost;
    
    % Backward propogation
    
    % Deltas required for gradient calculation
    d_3 = J(z_2,T); % 50X1
    d_2 = d_3*w2'.*h_tanh_derivative(z_1); %could be z_1 or a_1 most likely a_1 for relu only % Is this correct? % 50X3
    % Gradients for w1 and w2
    del1 = X_norm'*d_2; % 1X3 (w1 matrix gradients w.r.t cost) % THIS PART IS WRONG IN PART 1 !!! CHANGE a_1 to DATA! TRY AGAIN!
    del2 = z_1'*d_3; % 3X1 (w2 matrix gradients w.r.t cost)
    % Gradients for offset bias terms w1_b and w2_b
    del1_b1 = sum(d_2,1);
    del2_b2 = sum(d_3,1);
    
    V1 = V(Beta,V1,del1);
    w1 = W(w1,rho,V1);
    V2 = V(Beta,V2,del2);
    w2 = W(w2,rho,V2);
    V1_b = V(Beta,V1_b,del1_b1);
    w1_b = W(w1_b,rho,V1_b);
    V2_b = V(Beta,V2_b,del2_b2);
    w2_b = W(w2_b,rho,V2_b);
    
    
    % STOPPING CRITERIA (If "gradient" or "w(k+1) - w(k)" <= tolerance or iterations >= max_iter)
    
    gradient = [del1(:);del2(:);del1_b1(:);del2_b2(:)];
    w_norm = [w1(:);w2(:);w1_b(:);w2_b(:)];
    % Gradient = @(y,t,x,w_prev,lambda) x'*(y-t) + lambda.*w_prev;
    gradient_norm = norm(gradient); % Break out of while loop if gradient_norm < tolerance
    
    % Keep track of number of iterations
    niter = niter + 1;
    
    if niter >= max_iter
        fprintf('\nGradient Descent could not converge within max_iter: %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n', max_iter)
        break; % Break out of while loop
        
    elseif any(isnan(w_norm(:))) % If any of the weights become 'NaN' at any iteration
        fprintf('\nGradient Descent could not converge due to NaN values in w2 @ niter = %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n',niter)
        break; % Break out of while loop
        
    elseif gradient_norm < tolerance
        fprintf('\nGradient Descent converged to an answer due to norm(gradient) < tolerance with w_norm = %d\n',gradient_norm)
        break; % Break out of while loop
        
    end
    

    

end
toc;

% Plot cost vs iter#

figure()
title('Part 2.1 - Training Error');
xlabel('Epoch#');
ylabel('Training Error');
hold on
plot(1:niter,Total_Error);
legend('Training Error');
hold off


% Test data prediction
b1_test = ones(300,1)*w1_b; % 50X3
b2_test = ones(300,1)*w2_b; % 50X1

a_1_test = X_norm_test*w1 + b1_test;
z_1_test = h_tanh(a_1_test);
a_2_test = z_1_test*w2 + b2_test;
z_2_test = a_2_test;


% Plot data vs prediction
figure();
p3 = scatter(X,T,100,'filled','g');
hold on
p4 = plot(X_test,z_2_test,'r','LineWidth',2);
p5 = plot(X_test,T_actual,'b','LineWidth',2);
% Training Error at final iteration of gradient descent
Training_Error = Total_Error(end,:);
title(sprintf('Part 2.1 - Target vs Prediction (3 hidden units, Training Error: %d)',Training_Error));
xlabel('X');
ylabel('Y');
legend([p3,p5,p4],'Noisy Target','True model','Prediction');
hold off

% Pause before continuing to the next part
fprintf('\nPart 2.1 is completed! Please continue for Part 2.2 with 20 hidden units...\n');
pause;

%% Part 2 (Regression problem - Repeat with 20 hyperbolic tangen hidden units) 

% (One hidden layer composed of 20 hyperbolic tangent hidden units)


% Create data 
% rng(100) 
% X= linspace(-1,1,5000); % 2*rand(1,500)-1;
% randompermutation = randperm(size(X,2));
% X = X(:,randompermutation);
% % X = sort(X, 'descend');
% T=sin(2*pi*X)+0.3*randn(1,5000);
% X = X';
% X_norm = (X - mean(X))./std(X);
% T = T';

rng(100) 
X=2*rand(1,50)-1;
X = sort(X, 'descend');
T=sin(2*pi*X)+0.3*randn(1,50);
X = X';
X_norm = (X - mean(X))./std(X);
T = T';

% Create Test data 
X_test=linspace(-1,1,300);
X_test = sort(X_test, 'descend');
T_actual=sin(2*pi*X_test);
X_test = X_test';
X_norm_test = (X_test - mean(X))./std(X);
T_actual = T_actual';

% Hyperparameters
rho = 0.01;
lambda = 0; % This is not used here
Beta = 0.99;

% Gradient Descent stopping criteria
% Initial gradient norm to enter while loop (gradient descent)
gradient_norm = 1;
% Tolerance to stop iterations
tolerance = 1*10^-2;
% Maximum iterations to stop calculations
max_iter = 1000000;
% Iteration counter to track number of iterations
niter = 0;

% Initialize weights for each layer and bias terms at each layer
rng(100);
% Initial w (from input to hidden layer)
w1 = rand(1,20);
V1 = rand(1,20);
w1_b = rand(1,20); % Bias
V1_b = rand(1,20);
% Initial w (from hidden layer to output)
w2 = rand(20,1);
V2 = rand(20,1);
w2_b = rand(1,1); % Bias
V2_b = rand(1,1);

% Gradient Descent

% Track Cost vs niter
Total_Error = zeros(niter,1);
tic;
while gradient_norm > tolerance
    
    % Forward propogation
    
    % Calculate bias terms
    b_1 = ones(50,1)*w1_b; % 50X3
    b_2 = ones(50,1)*w2_b; % 50X1
    % Calculate the output
    a_1 = X_norm*w1 + b_1; % 50X3
    z_1 = h_tanh(a_1); % Using tanh activation function (input layer) % 50X3
    a_2 = z_1*w2 + b_2; % 50X1
    z_2 = a_2; % Using h(a) = a (output layer) % 50X1
    
    % Total cost
    Cost = Error_LS(T,z_2);
    fprintf('Iteration %d | Cost %d\n',niter,Cost);
    % Store cost at each iteration
    Total_Error(niter+1,1) = Cost;
    % Backward propogation
    % Deltas required for gradient calculation
    d_3 = J(z_2,T); % 50X1
    d_2 = d_3*w2'.*h_tanh_derivative(z_1); % Is this correct? % 50X3
    % Gradients for w1 and w2
    del1 = X_norm'*d_2; % 1X3 (w1 matrix gradients w.r.t cost) % THIS PART IS WRONG IN PART 1 !!! CHANGE a_1 to DATA! TRY AGAIN!
    del2 = z_1'*d_3; % 3X1 (w2 matrix gradients w.r.t cost)
    % Gradients for offset bias terms w1_b and w2_b
    del1_b1 = sum(d_2,1);
    del2_b2 = sum(d_3,1);
    
    %V = @(Beta,V,Gradient) Beta.*(V) + (1-Beta).*Gradient; % Velocity function (momentum)
    %W = @(w,rho,V) w - rho.*V; % Weight update using momentum

    V1 = V(Beta,V1,del1);
    w1 = W(w1,rho,V1);
    V2 = V(Beta,V2,del2);
    w2 = W(w2,rho,V2);
    V1_b = V(Beta,V1_b,del1_b1);
    w1_b = W(w1_b,rho,V1_b);
    V2_b = V(Beta,V2_b,del2_b2);
    w2_b = W(w2_b,rho,V2_b);
    
           
    % STOPPING CRITERIA (If "gradient" or "w(k+1) - w(k)" <= tolerance or iterations >= max_iter)
    
    gradient = [del1(:);del2(:);del1_b1(:);del2_b2(:)];
    w_norm = [w1(:);w2(:);w1_b(:);w2_b(:)];
    % Gradient = @(y,t,x,w_prev,lambda) x'*(y-t) + lambda.*w_prev;
    gradient_norm = norm(gradient); % Break out of while loop if gradient_norm < tolerance
    
    % Keep track of number of iterations
    niter = niter + 1;

    
    if niter >= max_iter
        fprintf('\nGradient Descent could not converge within max_iter: %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n', max_iter)
        break; % Break out of while loop
        
    elseif any(isnan(w_norm(:))) % If any of the weights become 'NaN' at any iteration
        fprintf('\nGradient Descent could not converge due to NaN values in w2 @ niter = %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n',niter)
        break; % Break out of while loop
        
    elseif gradient_norm < tolerance
        fprintf('\nGradient Descent converged to an answer due to norm(gradient) < tolerance with w_norm = %d\n',gradient_norm)
        break; % Break out of while loop
        
    end
    

    

end
toc;

% Plot cost vs iter#

figure()
title('Part 2.2 - Training Error');
xlabel('Epoch#');
ylabel('Training Error');
hold on
plot(1:niter,Total_Error);
legend('Training Error');
hold off

% Test data prediction
b1_test = ones(300,1)*w1_b; % 50X3
b2_test = ones(300,1)*w2_b; % 50X1

a_1_test = X_norm_test*w1 + b1_test;
z_1_test = h_tanh(a_1_test);
a_2_test = z_1_test*w2 + b2_test;
z_2_test = a_2_test;


% Plot data vs prediction

figure();
p3 = scatter(X,T,100,'filled','g');
hold on
p4 = plot(X_test,z_2_test,'r','LineWidth',2);
p5 = plot(X_test,T_actual,'b','LineWidth',2);
% Training Error at final iteration of gradient descent
Training_Error = Total_Error(end,:);
title(sprintf('Part 2.2 - Target vs Prediction (20 hidden units, Training Error: %d)',Training_Error));
xlabel('X');
ylabel('Y');
legend([p3,p5,p4],'Noisy Target','True model','Prediction');
hold off