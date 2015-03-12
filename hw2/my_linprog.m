function [x, fx, exitflag, output, lambda] = my_linprog(c, A, b, C, d)
options = optimset('Display', 'iter', 'LargeScale', 'off', 'Simplex', 'on');
[x, fx, exitflag, output, lambda] = linprog(c, A, b, C, d, [], [], [], options);
end

