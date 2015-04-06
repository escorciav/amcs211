clear all; close all; clc
% Define the domain of interest
x = linspace(-5, 5, 101);
a = 0;
m = [1, 2, 5, 10];
prm = {'-rs', '-gd', '-bv', '-mp'};
lgd = {'f(x) = e^x'};

% Plot f(x) = exp(x)
plot(x, abs(exp(x)- exp(x)), '-ko', 'linewidth', 2.0);
hold on;

% Plot f_m(x)
for i = 1:length(m)
  y = exp_taylor(x, m(i), a);
  z = abs((exp(x) - y)./exp(x));
  z(z>1) = 1;
  plot(x, z , prm{i}, 'linewidth', 1.5);
end
title(['Bounded relative error: abs( (e^x - f_m(x)) / e^x)']);
