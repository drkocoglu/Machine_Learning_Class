clear;
clc;
close all;


RBF = @(X,mu,sigma) exp(-(X-mu)./(2*sigma.^2));
% RBF = reshape(RBF,[1,11]);
RBF_NO = 11;
new_mu = linspace(0,1,RBF_NO);
sigma = 0.1;
x = [1;2;3;4;5;6];

for i = 1:RBF_NO
    rbf(:,i+1) = RBF(x,new_mu(i),sigma);
end

repeated_x = repmat(x,1,RBF_NO);
trial = ones(size(x,1),1);
rbf_test =[ones(size(x,1),1), RBF(repeated_x,new_mu,sigma)];
