function W = randInitializeWeights(L_in, L_out)
%Randomly initialize the weights of a layer with L_in incoming connections
%and L_out outgoing connections

W = zeros(L_out, 1 + L_in);

init_eps = 0.12;
W = rand(L_out, 1+L_in) * 2 * init_eps - init_eps; 

end
