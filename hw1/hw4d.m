clear all; close all; clc
% Define the domain of interest
x = linspace(-5, 5, 101);
a = 0;
m = [1, 2, 5, 10];
prm = {'-rs', '-gd', '-bv', '-mp'};
lgd = {'f(x) = e^x'};

% Plot f(x) = exp(x)
plot(x, exp(x), '-ko', 'linewidth', 2.0);
hold on;

% Plot f_m(x)
for i = 1:length(m)
  y = exp_taylor(x, m(i), a);
  plot(x, y, prm{i}, 'linewidth', 1.5);
  lgd = [lgd, {['f_m(x), m=', num2str(m(i))]}];
end
legend(lgd, 'location', 'NorthWest')
title(['Taylor expansion of e^x around a = ', num2str(a)]);