function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
temp = zeros(m,1);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Add ones to the X data matrix
Xcopy = [ones(m, 1) X];
% Activate second hidden layer
A2 = sigmoid(Xcopy*Theta1');
A2 = [ones(m,1),A2];
% Activate output layer
A3 = sigmoid(A2*Theta2');  


%%%%%%%%%%%%%%%%%% COST FUNCTION %%%%%%%%%%%%%%%%%%
for digit = 1:m
    % Recode the labels as vectors
    label = y(digit);
    y_vector = zeros(num_labels,1);
    if(label == 10)
       y_vector(10) = 1; 
    end
    y_vector(label) = 1;
    
    % first term in the cost function
    firstTerm = (-y_vector)' .* log(A3(digit,:));
    % first term in the cost function
    secondTerm = (1-y_vector)' .* (log(1-(A3(digit,:))));
    together = firstTerm - secondTerm;
    % sum over K labels and put in temp
    theSum = sum(together(:));
    temp(digit) = theSum;
    
end

% regularization term 
Theta1_Total = 0;
for i = 1:size(Theta1,1)
    for j = 2:size(Theta1,2)
        Theta1_Total = Theta1_Total + Theta1(i,j) ^2;
    end
end
Theta2_Total = 0;
for i = 1:size(Theta2,1)
    for j = 2:size(Theta2,2)
        Theta2_Total = Theta2_Total + Theta2(i,j) ^2;
    end
end

J = (sum(temp(:))/m) + lambda * (Theta1_Total + Theta2_Total) / (2 * m);


%%%%%%%%%%%%%%%%%% BACKPROP %%%%%%%%%%%%%%%%%%

% Difference between theta and error
Diff_1 = zeros(size(Theta1,1),size(Theta1,2));
Diff_2 = zeros(size(Theta2,1),size(Theta2,2));
for t = 1:m
    
    % Feedforward pass
    a_1 = [1 X(t,:)];
    a_2 = sigmoid(a_1*Theta1');
    a_2 = [1 a_2];
    a_3 = sigmoid(a_2*Theta2'); 
    
    % Recode the labels as vectors
    label = y(t);
    y_vector = zeros(num_labels,1);
    if(label == 10)
       y_vector(10) = 1; 
    end
    y_vector(label) = 1;
    
    % Compute "error term" delta for each layer
    delta_3 = a_3' - y_vector;
    delta_2 = Theta2' * delta_3 .* sigmoidGradient([1 a_1*Theta1']');
    
    % Accumulate the gradient from this example
    delta_2 = delta_2(2:end);
    
    Diff_2 = Diff_2 + delta_3 * a_2;
    Diff_1 = Diff_1 + delta_2 * a_1;  
    
end

% Regularization
Theta2_grad(:,1:1) = Diff_2(:,1:1) / m;
Theta2_grad(:,2:end) = (Diff_2(:,2:end) / m) + (lambda/m .* Theta2(:,2:end));

Theta1_grad(:,1:1) = Diff_1(:,1:1) / m;
Theta1_grad(:,2:end) = (Diff_1(:,2:end) / m) + (lambda/m .* Theta1(:,2:end));





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
