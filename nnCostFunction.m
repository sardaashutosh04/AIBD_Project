%Cost function which calculates the cost and the gradients of the neural
%network given the parameters, X and y. 
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% Reshaping nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for the 2 layer neural network
theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
X = [ones(m,1) X];        
 
J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));
%Changing labels to output vector (Ex. label = 5 will be changed
%to [0,0,0,0,1,0,0,0,0,0])
y_expand = zeros(m, num_labels);
for i = 1:m
    k = y(i);
    y_expand(i,k) = 1.0;
end
% Forward Processing
z2 = X*theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2*theta2';
h = sigmoid(z3);

%Calculating Cost
j = (1/m)*sum(sum(((-y_expand).*log(h))-((1-y_expand).*log(1-h))));
reg_theta1 = theta1; reg_theta1(1) = 0.0;
reg_theta2 = theta2; reg_theta2(1) = 0.0;
J = j + (lambda/(2*m))*(sum(sum(reg_theta1.^2))+sum(sum(reg_theta2.^2)));

%Backward Processing
d3 = h - y_expand;
d2 = (d3*theta2);
d2(:,1) = [];
d2 = d2.*sigmoidGradient(z2);

%Calculating Gradients
theta1_grad = (1/m)*(theta1_grad + d2'*X) + (lambda/m)*reg_theta1;
theta2_grad = (1/m)*(theta2_grad + d3'*a2) + (lambda/m)*reg_theta2;

% Unrolling gradients
grad = [theta1_grad(:) ; theta2_grad(:)];


end
