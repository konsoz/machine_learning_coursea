function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%1

% Cost function
% This is BAD solution and it should be vectorized.
%
% You can use the R matrix to set selected entries to 0.
% For example, R .* M will do an element-wise multiplication between M
% and R; since R only has elements with values either 0 or 1, this has the
% elect of setting the elements of M to 0 only when the corresponding value
% in R is 0. Hence, sum(sum(R.*M)) is the sum of all the elements of M for
% which the corresponding element in R equals 1.
for i= 1:size(Y,1)
   for j = 1:size(Y,2)
     if R(i,j) == 1
        result = ((Theta(j,:)*X(i,:)')-Y(i,j)).^2;
        J = J + result;
     end
   end
end

J = J*0.5;
J = J + lambda*sum(sum(Theta.^2))/2 + lambda*sum(sum(X.^2))/2; 

% Gradient
for i= 1:num_movies
    % users that have rated movie i.
    userHasRatedIdx = find(R(i, :)==1);
     % user features of movie i.
    Theta_temp = Theta(userHasRatedIdx,:);
    % user's ratings of movie i.
    Y_temp = Y(i,userHasRatedIdx);
    % Gradient for x
    X_grad(i, :) = (X(i,:)*Theta_temp' - Y_temp)*Theta_temp;
    % Regularized x
    X_grad(i, :) = X_grad(i, :)+lambda*X(i, :);
    
end
 
for j= 1:num_users
    % movies that have rated by user j
    idx = find(R(:,j)==1);
    % features of movies rated by user j.
    X_tmp = X(idx, :);
    % user ratings by user j.
    Y_temp = Y(idx,j);
    % theta gradient
    Theta_grad(j, :) = (X_tmp*Theta(j,:)' - Y_temp)'*X_tmp;
    % Regularized Theta
    Theta_grad(j, :) = Theta_grad(j, :)+ lambda*Theta(j,:);
end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
