function gamma = geometric_margin(G, y, w, b)

    if nargin < 4 || isempty(b)
        b = 0;
    end
    w = w(:);
    denom = norm(w);
    if denom == 0
        gamma = NaN;
        return
    end
    s = y(:) .* (G * w + b);
    gamma = min(s) / denom;
end
