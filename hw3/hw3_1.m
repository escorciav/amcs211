function hw3_1()
[t, y] = ex_data();
sol = struct('l1', [0, 0], 'l2', [0, 0], 'linf', [0, 0]);

[y_l1, sol.l1] = l1_fitting(t, y);
[y_l2, sol.l2] = l2_fitting(t, y);
[y_linf, sol.linf] = linf_fitting(t, y);
plot_regression(t, y, [y_l1, y_l2, y_linf]);
plot_function(t, y, [sol.l1 sol.l2 sol.linf]');
end

function [y, x] = l2_fitting(A, b)
% Perfrom a l2-linear regression of A plus a bias term onto b
m = size(A, 1);
A = [A, ones(m, 1)];
x = A \ b;
y = A*x;
end

function [y, x] = l1_fitting(A, b)
% Perfrom a l1-linear regression of A plus a bias term onto b
[m, n] = size(A);
n = n + 1;
A = [A, ones(m, 1)];
c = [zeros(n, 1); ones(m, 1)];
At = [ A, -eye(m);
      -A, -eye(m)];
bt = [b; -b];
x_lin = linprog(c, At, bt);
x = x_lin(1:2);
y = A*x;
end

function [y, x] = linf_fitting(A, b)
% Perfrom a linf-linear regression of A plus a bias term onto b
[m, n] = size(A);
n = n + 1;
A = [A, ones(m, 1)];
c = [zeros(n, 1); ones(1, 1)];
At = [ A, -ones(m, 1);
      -A, -ones(m, 1)];
bt = [b; -b];
x_lin = linprog(c, At, bt);
x = x_lin(1:2);
y = A*x;
end

function plot_function(t, y, X)
reg = {'data', 'l_1', 'l_2', 'l_\infty'};
colors = 'bgr';
figure;
plot(t, y, 'k.');
hold on;
T = [min(t) 1;max(t) 1];
for i = 1:size(X, 1)
  u = T * X(i, :)';
  line(T(:, 1), u, 'Color', colors(i))
end
legend(reg(1:size(X, 1)+1));
xlabel('t')
title('Plot of linear functions obtained with different norms on residuals')
end

function plot_regression(t, y, y_est)
reg = {'data', 'l_1', 'l_2', 'l_\infty'};
plot(t, y, 'k.');
hold on;
plot(t, y_est, 's');
legend(reg(1:size(y_est, 2)+1));
xlabel('t')
title('Plot of Regressions using different norms on residuals')
end