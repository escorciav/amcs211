import sympy

x1, x2 = sympy.symbols('x1 x2')
f = 100*(x2 - x1**2)**2 + (1-x1)**2

df_dx1 = sympy.diff(f,x1)
df_dx2 = sympy.diff(f,x2)
H = sympy.hessian(f, (x1, x2))

xs = sympy.solve([df_dx1, df_dx2], [x1, x2])

H_xs = H.subs([(x1,xs[0][0]), (x2,xs[0][1])])

flag = True
for i in H_xs.eigenvals().keys():
    if i.evalf() <= 0:
        flag = False
        break

if flag:
    print 'Stationary point'
else:
    print 'Saddle point'
