clear; clc;

X = [
    18  2 46 107   0  0  1
     8  0  7   6   0  0  0
    18  0  1   4   2  0  0
    12  7 19 189   0  0  0
    34  0  0   3   1  0  0
     1  1 62   0  38  1  6
     0  1  4   0   7  1  3
     0  3  1   0  15 12 19
     1  1  2   0 326  5 11
     9 24  0   3 177  8 26
];

y = [ones(5, 1); -ones(5, 1)];

eps_norm = 1e-6;
row_norm = sqrt(sum(X.^2, 2)) + eps_norm;
X_tilde = X ./ row_norm;
mu_X = mean(X_tilde, 1);
G = X_tilde - mu_X;

[N, d] = size(G);

% --- (a) Perceptron ---
T = 1000;
w_p_row = perceptron(G, y, T);
w_p = w_p_row(:);

gamma_p = geometric_margin(G, y, w_p, 0);

fprintf('(a) Perceptron\n');
fprintf('    w_p^T = [');
fprintf('%.3f ', w_p');
fprintf(']\n');
fprintf('    gamma_p = %.3f\n\n', gamma_p);

% --- (b) Linear SVM  ---

cvx_begin quiet
    variables w_s(d) b_s
    minimize( 0.5 * (w_s' * w_s) )
    subject to
        y .* (G * w_s + b_s) >= 1;
cvx_end

gamma_s = geometric_margin(G, y, w_s, b_s);

fprintf('(b) Linear SVM \n');
fprintf('    w_s^T = [');
fprintf('%.3f ', w_s');
fprintf(']\n');
fprintf('    b_s = %.3f\n', b_s);
fprintf('    gamma_s = %.3f\n\n', gamma_s);

% --- (c) Gaussian Bayes, equal priors 1/2, ridge on covariances ---
lambda_ridge = 1e-3;
fprintf('(c) Gaussian Bayes (ridge lambda = %.3f)\n', lambda_ridge);

idx_pos = y == 1;
idx_neg = y == -1;
G_pos = G(idx_pos, :);
G_neg = G(idx_neg, :);

mu_plus = mean(G_pos, 1)';
mu_minus = mean(G_neg, 1)';

%regularization
Sigma_plus = cov(G_pos, 1) + lambda_ridge * eye(d);
Sigma_minus = cov(G_neg, 1) + lambda_ridge * eye(d);

fprintf('    mu_+ =\n    ');
fprintf('%.3f  ', mu_plus');
fprintf('\n');
fprintf('    Sigma_+ =\n');
print_matrix_3dp(Sigma_plus);
fprintf('    mu_- =\n    ');
fprintf('%.3f  ', mu_minus');
fprintf('\n');
fprintf('    Sigma_- =\n');
print_matrix_3dp(Sigma_minus);

log_prior = log(0.5);
y_hat = zeros(N, 1);
for i = 1:N
    g = G(i, :)';
    ll_plus = log_prior + logmvnpdf_col(g, mu_plus, Sigma_plus);
    ll_minus = log_prior + logmvnpdf_col(g, mu_minus, Sigma_minus);
    if ll_plus >= ll_minus
        y_hat(i) = 1;
    else
        y_hat(i) = -1;
    end
end

train_err = mean(y_hat ~= y);
fprintf('    Training error = %.3f (%d / %d misclassified)\n', train_err, sum(y_hat ~= y), N);

% --- Local helper: print matrix with 3 decimal places ---
function print_matrix_3dp(M)
    [nr, nc] = size(M);
    for r = 1:nr
        fprintf('    ');
        fprintf('%.3f  ', M(r, :));
        fprintf('\n');
    end
end

% --- Local helper: log-density of N(mu, Sigma), x and mu column vectors ---
function lp = logmvnpdf_col(x, mu, Sigma)
    dloc = length(x);
    xc = x - mu;
    R = chol(Sigma, 'lower');
    log_det = 2 * sum(log(diag(R)));
    z = R \ xc;
    quad = z' * z;
    lp = -0.5 * (dloc * log(2 * pi) + log_det + quad);
end
