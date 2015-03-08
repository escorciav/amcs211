function [x, fx] = my_linprog(c, A, b, C, d)
options = optimset('Display', 'iter');
x = linprog(c, A, b, C, d, [], [], [], options);
fx = c'*x;
end

