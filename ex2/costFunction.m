function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
A = transpose(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
z = 0;
h = 0;
for i = 1 : m
    h = sigmoid(A * transpose(X(i,:)));
    z = z + (-y(i) * log(h) - (1 - y(i)) * log(1 - h));
end

J = z / m;

for j = 1 : size(grad)
    w = 0;
    for i = 1 : m
        h = sigmoid(A * transpose(X(i,:)));
        w = w + (h - y(i)) * X(i,j);
    end
    grad(j) = grad(j) + w / m;
end






% =============================================================

end
