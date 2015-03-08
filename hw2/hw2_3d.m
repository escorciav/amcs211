% HW2: Linear Programming - P3d
% Author: Victor Escorcia
% Date: March 2015
c = [0; -1; -1; 1; 0];
A = [0, 2, 2, 2, -4;
     0, -2, 2, -2, 0;
     4, 0, 2, 0, 2];
b = [2; 2; 2];
C = [eye(size(A,2)); -eye(size(A,2))];
d = ones(size(C,1), 1);

[x_p, f_x] = my_linprog(c, C, d, A, b);
disp('Optimum:')
disp(f_x)
disp('Minimizer:')
disp(x_p)
