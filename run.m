
%% Initialization

clear ; close all; clc

%% Setup the parameters you will use for this exercise

input_layer_size  = 784;  
hidden_layer_size = 500;   % number of hidden units
num_labels = 10;          % 10 classes(labels), from 0 to 9   
                          
%% =========== Part 1: Loading Data =============
% Load Training Data
fprintf('Loading Data ...\n')

load('mnist_data.mat');
X = trainImages;
y = trainLabels';
test_X = testImages;
test_y = testLabels';
m = size(X, 1);
X = cast(X, 'double');
test_X = cast(test_X, 'double');
y = y_change(y);                % (note that I have mapped "0" to label 10)
test_y = y_change(test_y);
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Initializing Parameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Part 3: Training Neural Network ===================

fprintf('\nTraining Neural Network... \n')

%Hyper Parameters
lambda = 30; %Regularization parameter that prevents over fitting
lr = 0.5; %Learning Rate
alpha = 0.3; %Momentum
epoch = 150; %Number of Iterations

%Gradient Decsent
vel = zeros(size(nn_params));
acc = zeros(epoch,1);
test_acc = zeros(epoch,1);
loss = zeros(epoch, 1);
test_loss = zeros(epoch, 1);
for i = 1:epoch
    %Randomly shuffling the data
    train = [X y];
    train = train(randperm(size(train,1)),:);
    X = train(:,(1:784));
    X = cast(X, 'double');
    y = train(:,785);
    
    %Computing cost and gradient
    %for training data
    [cost, grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda);
    %for test data
    [test_cost, test_grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    test_X, test_y, lambda);
    
    loss(i) = cost;
    test_loss(i) = test_cost;
    
    %Updating nn_parameters(weights)
    vel = alpha*vel - (lr.*grad);
    nn_params = nn_params + vel;
    
    fprintf('\nEpoch: ');
    disp(i);
    fprintf('\nCost: ');
    disp(cost);
    
    % Obtaining Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
    pred = predict(Theta1, Theta2, X);
    acc(i) = mean(double(pred==y));
    test_pred = predict(Theta1, Theta2, test_X);
    test_acc(i) = mean(double(test_pred==test_y));
end


fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 4: Implementing Predict and Plotting Accuracy amd Loss =================

%Printing accuracies
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nTest Set Accuracy: %f\n', mean(double(test_pred == test_y)) * 100);

%Plotting accuracy vs epochs for training and test sets
figure
plot((1:epoch), acc)
hold on
plot((1:epoch), test_acc)
title('Accuracy vs Epoch')
xlabel('Epoch')
ylabel('Accuracy')
legend({'y = Training Set','y = Test Set'},'Location','southeast')
hold off

%Plotting loss vs epochs for training and test sets
figure
plot((1:epoch), loss)
hold on
plot((1:epoch), test_loss)
title('Loss vs Epoch')
xlabel('Epoch')
ylabel('Loss')
legend({'y = Training Set','y = Test Set'},'Location','northeast')
hold off
