function y = sigmoidGradient(x)
    y = sigmoid(x) .* (1-sigmoid(x)); %Calculates the derevative of sigmoid function
end