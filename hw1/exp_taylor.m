function y = exp_taylor(x, m, a)
% Compute the Taylor expansion of ordern m for exp(x) around the point a
y = zeros(size(x));
for n = 0:m
  y = y + exp(a)*(1/factorial(n))*(x-a).^n;
end
end