close all;
clear all;

% Load data
D = dload('questionnaire_data.txt');

% Setup design matrix
% Intercept term will estimate GroupA effects
X = [D.GroupB D.GroupC D.GroupD]; 

% Estimate Degrees of Freedom
p = 4;
n = length(D.Group);
DFM = p - 1;
DFE = n - p;

% Run a regression for group effects each day;
for d = 1:5
    eval(sprintf('Y = D.binDay_%d;',d));
    
    [B, DEV, STATS] = glmfit(X, Y, 'binomial', 'link', 'probit');
    y_hat = glmval(B, X, 'probit');
    
    SST = sum((Y-mean(Y)).^2);
    SSE = sum(STATS.resid.^2);
    MSE = SSE/DFE;
    
    SSM = sum((y_hat - mean(Y)).^2);
    MSM = SSM/DFM;
    
    F(d) = MSM/MSE;
    P(d) = 1-fcdf(F(d), DFM, DFE);
    
end;