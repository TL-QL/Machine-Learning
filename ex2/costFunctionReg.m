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
A = transpose(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z = 0;
h = 0;
for i = 1 : m
    h = sigmoid(A * transpose(X(i,:)));
    z = z + (-y(i) * log(h) - (1 - y(i)) * log(1 - h));
end

t = 0;
for k = 2 : size(theta)
    t = t + theta(k) ^ 2;
end

J = z / m + lambda * t / (2 * m);

w = 0;
for i = 1 : m
    h = sigmoid(A * transpose(X(i,:)));
    w = w + (h - y(i)) * X(i,1);
end
grad(1) = grad(1) + w / m;
    
for j = 2 : size(grad)
    w = 0;
    for i = 1 : m
        h = sigmoid(A * transpose(X(i,:)));
        w = w + (h - y(i)) * X(i,j);
    end
    grad(j) = grad(j) + w / m + lambda * theta(j) / m;
end






% =============================================================

end
