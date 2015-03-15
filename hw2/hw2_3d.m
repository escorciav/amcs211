% HW2: Linear Programming - P3d
% Author: Victor Escorcia
% Date: March 2015
A = [0, 2, 2, 2, -4;
     0, -2, 2, -2, 0;
     4, 0, 2, 0, 2];
b = [2; 2; 2];
C = [eye(size(A,2)); -eye(size(A,2))];
d = ones(size(C,1), 1);

x = [0; 1; 1; -1; 0];
J = find((d - C*x) == 0);
c = -(A'*[1; 1; 1] + C(J, :)'*[1; 1; 1])

[x_p, f_x] = my_linprog(c, C, d, A, b);
disp('Optimum:')
f_x
disp('Minimizer:')
x_p
disp(['x_p == x: ', num2str(all(x == x_p))])
