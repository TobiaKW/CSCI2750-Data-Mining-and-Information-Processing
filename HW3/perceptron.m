function w = perceptron(X, y, T)

    [N, d] = size(X);
    w = zeros(1, d);
    iter = 1;

    while iter <= T
        found = false;
        for i = 1:N
            if y(i) * (w * X(i, :)') <= 0
                w = w + y(i) * X(i, :);
                found = true;
                break;
            end
        end

        if ~found
            break;
        end
        iter = iter + 1;
    end
end
