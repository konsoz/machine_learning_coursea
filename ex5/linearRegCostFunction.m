function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


n = size(X,2);
dif = X*theta - y;
% Fill theta1 with zeros, bias should not be used in regularization
theta1 = [0 ; theta(2:size(theta), :)];
thetaPowerOf2 = theta1.^2;
% Regularization term for cost function
regularization = lambda/(2*m) * sum(thetaPowerOf2);
% The cost function
J = (dif'*dif)/(2*m) + regularization;
% Gradient 
grad = (X' * dif + (lambda*theta1))/m;

% =========================================================================

grad = grad(:);

end
