function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
possible_C = [0.01 0.03 0.1 0.3 1 3 10 30]';
possible_sigma = [0.01 0.03 0.1 0.3 1 3 10 30]';
C = 0;
sigma = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


currentError = intmax('int64');

for i=1:length(possible_C)
    currentC = possible_C(i);
    for j=1:length(possible_sigma)
        current_sigma = possible_sigma(j);
        model= svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
        predictions = svmPredict(model,Xval);
        error = mean(double(predictions ~= yval));
        if error < currentError
            currentError = error;
            C = currentC;
            sigma = current_sigma;
        end
    end
end




% =========================================================================

end
