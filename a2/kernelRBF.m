
% Clear variables and close figures
clear all; close all;

% Load data
load nonLinear.mat % Loads {X,y,Xtest,ytest} [n,d] = size(X);
[t, ~] = size(Xtest);
% Find best value of RBF kernel parameter ,
% training on 9/10 of the train set and validating on the remaining
minErr = inf;
nSplits = 5;
for sigma = 2.^[ -15:15]
    for lambda = 2.^[ -15:15] validError = 0;
        for split = 1: nSplits
% Get the training set and test set indices
testStart = 1 + (n/nSplits )*( split -1); testEnd = (n/nSplits)*split;
trainNdx = [1: testStart-1 testEnd+1:n]; testNdx = testStart : testEnd ;
% Train on the training set
model = leastSquaresRBFL2(X(trainNdx ,:) ,y(trainNdx),lambda,sigma);
% Compute the error on the validation set
yhat = model.predict(model,X(testNdx));
validError = validError + sum((yhat - y(testNdx)).^2)/(n/nSplits ); end
fprintf('Error with lambda = %.3e, sigma = %.3e = %.2f\n',lambda,sigma,validError);
% Keep track of the lowest validation error
if validError < minErr minErr = validError ; bestLambda = lambda; bestSigma = sigma;
        end
    end
end
fprintf( 'Value of lambda and sigma that achieved the lowest validation error were %.3e and
% Train least squares model on training data
model = leastSquaresRBFL2(X,y,bestLambda,bestSigma);
% Test least squares model on test data
yhat = model.predict(model,Xtest);
% Report test error
squaredTestError = sum(( yhat-ytest ).^2)/ t % Plot model
figure (1);
plot(X,y, 'b. ');
hold on
plot(Xtest , ytest , 'g. ' );
Xhat = [min(X):.1:max(X)] ; % Choose points to evaluate the function yhat = model.predict(model,Xhat);
plot(Xhat,yhat, 'r');
ylim([-300 400]);
print ?dpng nonLinear2 . png