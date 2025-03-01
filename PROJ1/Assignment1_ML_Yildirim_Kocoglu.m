% Assignment#1 - Machine Learning
% Name: Yildirim Kocoglu

% Clear and close all
clear;
clc;
close all;

% Load the data
Data = xlsread('C:\Users\ykocoglu\Desktop\Classes\ML TTU\PROJ1\proj1Dataset.xlsx', 1, 'A2:B407'); % Please change the directory path (if needed)

% Delete any "Nan rows" (Missing Data)
Data(any(isnan(Data), 2), :) = [];

% Seperate the Data into x (features - Weight) and y (realizations - Horsepower)
x = Data(:,1);
y = Data(:,2);

% Plot the scatter plot of x vs y
figure(1)
scatter(x,y,50,'x','r','LineWidth',2);
title('Matlab''s "carbig" dataset');
xlabel('Weight');
ylabel('Horsepower');
hold on;
%%  Analytical solution


x = [Data(:,1), ones(size(Data,1),1)];

% Analytical solution - equation
w = pinv(x'*x)*x'*y;
y1 = x*w;

% Cost function from analytical solution
JJ = sum((y - y1).^2);

% Plot the analytical solution line on the scatter plot

p = plot(x(:,1),y1, 'LineWidth',2, 'color', 'b');
legend(p,'Closed form')
hold off;

%%  Gradient Descent solution

% Set learning rate (rho) - CAN BE ADJUSTED!
rho = 0.001;

% Set Random initial guess between 0 and 1 for weights (changes at each run!)
w2 = rand(2,1);


% Initialize niter to keep track in the while loop
niter = 0;

% Initialize Cost function (J) to monitor
J = zeros(niter,1);
w_matrix = zeros(size(w2,1),niter);

% Normalize the data
xnorm = (x(:,1)-mean(x(:,1)))/std(x(:,1));

% Add 1 vector to xnorm for bias
xnorm = [xnorm,ones(size(xnorm,1),1)];

% Set initial gradient_norm and tolerance in the following scale:
% gradient_norm > tolerance (To avoid issues within the while loop) - CAN BE ADJUSTED!
gradient_norm = 1;
tolerance = 1*10^-1;

% Set maximum number of iterations for the while loop - CAN BE ADJUSTED!
max_iter = 10000;

% GRADIENT DESCENT ALGORITHM!
while gradient_norm > tolerance

% Store the weights (w)
w_matrix(:,niter+1) = w2;
gradient = (2.*w2'*(xnorm'*xnorm) - 2.*y'*xnorm);
w2 = w2 - rho.*gradient';

% Store Cost function (J)
y2 = xnorm*w2;
J(niter+1) = sum((y - y2).^2);

% Keep track of number of iterations
niter = niter + 1;

% STOPPING CRITERIA (If gradient <= tolerance or iterations >= 10000)

gradient_norm = norm(gradient); % Break out of while loop if gradient_norm < tolerance

if niter >= max_iter
    fprintf('Gradient Descent could not converge within max_iter: %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n', max_iter)
    break; % Break out of while loop
    
elseif any(isnan(w2)) % If any of the weights become 'NaN' at any iteration
    fprintf('Gradient Descent could not converge due to NaN values in w2 @ niter = %d \n\nTry to adjust the following parameters in the following order:\n1) rho\n2) max_iter\n3) tolerance ( < initial gradient_norm)\n\nTERMINATING THE PROGRAM!\n',niter)
    break; % Break out of while loop
end

end % End of while loop

% Printf number of iterations it took to converge (gradient descent)
if niter < max_iter && ~any(isnan(w2))
   fprintf('Number of iterations: %d\n', niter)
end

% Convert weights obtained from gradient descent with normalized data to
% fit the un-normalized data (Not sure if necessary!)
w2 = x\(xnorm*w2);
y3 = x*w2;

% Plot Gradient Descent solution line on the scatter plot of data
figure(2)
x = Data(:,1);
y = Data(:,2);
scatter(x,y,50,'x','r','LineWidth',2);
title('Matlab''s "carbig" dataset');
xlabel('Weight');
ylabel('Horsepower');
hold on;
p5 = plot(x(:,1),y3, 'LineWidth',2, 'color', 'g');
legend(p5,'Gradient Descent')
hold off;

% Plot & Monitor the Cost function (J) (OPTIONAL!)

option = 'on'; % Choose 'on' or 'off' to turn on\off plot - CAN BE ADJUSTED! 

if strcmp(option, 'on')
figure(3)
title('OPTIONAL PLOT: J(w) vs niter');
xlabel('niter');
ylabel('J(w)');
hold on
p2 = plot(1:niter,J, 'LineWidth',3, 'color', 'b');
p3 = scatter(1,JJ,'x','g','LineWidth',10);
p4 = scatter(niter,J(:,niter),'x','r','LineWidth',10);
legend([p2, p3, p4],'Gradient Descent J(w)', 'Analytical solution optimum J(w)', 'Gradient Descent optimum J(w)')
hold off
elseif strcmp(option, 'off')
    fprintf('J(w) vs niter is not plotted\nChange option to plot!\n');
else
    error('Please choose "on" or "off" to turn on/off plot. CHECK THE SPELLING!'); 
end