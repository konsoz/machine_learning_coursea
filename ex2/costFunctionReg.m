function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = size(X,2); % number of features + 1 (for constant term)
hypothes = sigmoid(X*theta);

firstTerm = (-y)'*log(hypothes);
secondTerm = (1-y)'*log(1-hypothes);

theta1 = theta(2:n,:);

thetaPowerOf2 = theta1.^2;

regularization = lambda/(2*m) * sum(thetaPowerOf2);

J = (1/m) * sum(firstTerm-secondTerm) + regularization;

 E = hypothes - y;
 theSum = sum(E.*X);
 
 temp = (1/m) * theSum;
 
 grad(1,1) = temp(1,1);
 
 for index = 2:size(grad,1)
     grad(index) = ((1/m) * theSum(index)) + (lambda/m) * theta(index);
 end
 
% =============================================================

end
