import sympy

x1, x2 = sympy.symbols('x1 x2')
f = 100*(x2 - x1**2)**2 + (1-x1)**2

df_dx1 = sympy.diff(f,x1)
df_dx2 = sympy.diff(f,x2)
H = sympy.hessian(f, (x1, x2))

xs = sympy.solve([df_dx1, df_dx2], [x1, x2])

H_xs = H.subs([(x1,xs[0][0]), (x2,xs[0][1])])
lambda_xs = H_xs.eigenvals()

count = 0
for i in lambda_xs.keys():
    if i.evalf() <= 0:
        count += 1

if count == 0:
    print 'Local minima'
elif count == len(lambda_xs.keys()):
    print 'Lacal maxima'
else:
    print 'Saddle point'
