function y = y_change(y)
for i = 1:size(y,1)
    if y(i) == 0
        y(i) = 10; %I'm changing all "0" to "10" because MATLAB indexing starts from 1 and not 0
    end
end
end
